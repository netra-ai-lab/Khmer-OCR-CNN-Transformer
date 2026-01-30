import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from config import OCRConfig
from tokenizer import Tokenizer
from preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

class OCRPredictor:
    def __init__(self, 
                 model_path: str | Path, 
                 tokenizer: Tokenizer, 
                 config: OCRConfig, 
                 model_class):
        
        self.cfg = config
        self.tokenizer = tokenizer
        self.device = torch.device(self.cfg.device)
        self.preprocessor = ImagePreprocessor(config)

        # Initialize Architecture
        logger.info(f"Init Model: dim={self.cfg.emb_dim}, max_seq={self.cfg.max_seq_len}")
        self.model = model_class(
            vocab_size=len(tokenizer),
            pad_idx=tokenizer.pad_idx,
            emb_dim=self.cfg.emb_dim,
            max_global_len=self.cfg.max_seq_len
        )

        # Load Weights
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            logger.warning("Strict load failed. Retrying with strict=False")
            self.model.load_state_dict(state_dict, strict=False)

    def predict(self, image_input, beam_width: int = 3) -> str:
        chunks = self.preprocessor.process(image_input).to(self.device)
        
        with torch.no_grad():

            f = self.model.cnn(chunks)
            
            p_out = self.model.patch(f)
            p = p_out[0] if isinstance(p_out, tuple) else p_out
            p = p.transpose(0, 1).contiguous()
            
            enc_out = self.model.enc(p).transpose(0, 1)

            N, L, D = enc_out.shape
            merged = enc_out.reshape(1, N * L, D)

            B, T, _ = merged.shape
            limit = min(T, self.model.global_pos.size(0))
            pos_emb = self.model.global_pos[:limit, :].unsqueeze(0)
            
            if T > limit: 
                merged = merged[:, :limit, :] + pos_emb
            else: 
                merged = merged + pos_emb

            # BiLSTM Smoothing Check
            if hasattr(self.model, 'context_bilstm'):
                self.model.context_bilstm.flatten_parameters()
                memory, _ = self.model.context_bilstm(merged)
            else:
                memory = merged

            if beam_width <= 1:
                return self._greedy_decode(memory)
            else:
                return self._beam_search(memory, beam_width)

    def _greedy_decode(self, memory):
        B, T, _ = memory.shape
        mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        generated = [self.tokenizer.sos_idx]

        for _ in range(self.cfg.decode_max_len):
            tgt = torch.LongTensor([generated]).to(self.device)
            logits = self.model.dec(tgt, memory, mask)
            next_token = torch.argmax(logits[0, -1, :]).item()
            
            if next_token == self.tokenizer.eos_idx:
                break
            generated.append(next_token)
            
        return self.tokenizer.decode(generated)

    def _beam_search(self, memory, beam_width):
        B, T, D = memory.shape
        memory = memory.expand(beam_width, -1, -1)
        mask = torch.zeros((beam_width, T), dtype=torch.bool, device=self.device)
        
        beams = [(0.0, [self.tokenizer.sos_idx])]
        completed = []

        for _ in range(self.cfg.decode_max_len):
            k_curr = len(beams)
            current_seqs = [b[1] for b in beams]
            tgt = torch.tensor(current_seqs, dtype=torch.long, device=self.device)
            
            logits = self.model.dec(tgt, memory[:k_curr], mask[:k_curr])
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            candidates = []
            for i in range(k_curr):
                score, seq = beams[i]
                top_probs, top_idxs = log_probs[i].topk(beam_width)
                for k in range(beam_width):
                    candidates.append((score + top_probs[k].item(), seq + [top_idxs[k].item()]))

            candidates.sort(key=lambda x: x[0], reverse=True)
            next_beams = []
            for s, seq in candidates:
                if seq[-1] == self.tokenizer.eos_idx:
                    completed.append((s / len(seq), seq))
                elif len(next_beams) < beam_width:
                    next_beams.append((s, seq))
            
            beams = next_beams
            if not beams: break

        best_seq = sorted(completed, key=lambda x: x[0], reverse=True)[0][1] if completed else beams[0][1]
        return self.tokenizer.decode(best_seq)