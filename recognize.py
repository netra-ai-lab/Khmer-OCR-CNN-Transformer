import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
import sys

# Import your model architecture
# from vgg_model import KhmerOCR 
# from resnet_model import KhmerOCR
from crnn_se_model import KhmerOCR

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def chunk_image_inference(img_tensor, chunk_width=100, overlap=16):
    """
    Splits the tensor into overlapping chunks to handle long text lines.
    """
    C, H, W = img_tensor.shape
    chunks = []
    start = 0

    while start < W:
        end = min(start + chunk_width, W)
        chunk = img_tensor[:, :, start:end]

        # Pad last chunk if shorter than chunk_width (Pad with 1.0 = White)
        if chunk.shape[2] < chunk_width:
            pad_size = chunk_width - chunk.shape[2]
            chunk = F.pad(chunk, (0, pad_size, 0, 0), value=1.0)

        chunks.append(chunk)
        start += chunk_width - overlap

    return chunks

# ==============================================================================
# 2. INFERENCE CLASS
# ==============================================================================

class KhmerOCRInference:
    def __init__(self, 
                 model_path, 
                 char2idx_input, 
                 model_class,      # Pass the class definition (KhmerOCR)
                 emb_dim=384,      # 256 for your new models
                 device='cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 1. Load Vocabulary
        if isinstance(char2idx_input, dict):
            self.char2idx = char2idx_input
        elif isinstance(char2idx_input, str):
            with open(char2idx_input, 'r', encoding='utf-8') as f:
                self.char2idx = json.load(f)
        else:
            raise ValueError("char2idx_input must be a dictionary or a json file path.")

        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.sos_idx = self.char2idx.get("<sos>")
        self.eos_idx = self.char2idx.get("<eos>")
        self.pad_idx = self.char2idx.get("<pad>", 0)

        # 2. Initialize Model
        self.model = model_class(
            vocab_size=len(self.char2idx),
            pad_idx=self.pad_idx,
            emb_dim=emb_dim,       
            max_global_len=4096
        )

        # 3. Load Weights
        print(f"⏳ Loading weights from {model_path} (Dim: {emb_dim})...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Load weights
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"⚠️ Strict loading failed (likely due to missing/extra keys). Retrying with strict=False...")
            self.model.load_state_dict(state_dict, strict=False)
            
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(), 
        ])

    def preprocess(self, image_input):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            image = Image.open(image_input).convert('L')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('L')
        else:
            raise ValueError("Input must be a file path or PIL Image.")

        target_height = 48
        aspect_ratio = image.width / image.height
        new_width = int(target_height * aspect_ratio)
        new_width = max(10, new_width) 
        
        image = image.resize((new_width, target_height), Image.Resampling.BILINEAR)
        img_tensor = self.transform(image) 
        
        # Helper function usage (ensure chunk_image_inference is defined)
        chunks = chunk_image_inference(img_tensor, chunk_width=100, overlap=16)
        chunks_norm = [(c - 0.5) / 0.5 for c in chunks]
        
        return torch.stack(chunks_norm).to(self.device)

    def encode_image(self, chunks_tensor):
        with torch.no_grad():
            # 1. CNN
            f = self.model.cnn(chunks_tensor)
            
            # 2. Patch
            patch_out = self.model.patch(f)
            p = patch_out[0] if isinstance(patch_out, tuple) else patch_out

            # 3. Encoder
            p = p.transpose(0, 1).contiguous()
            enc_out = self.model.enc(p)
            enc_out = enc_out.transpose(0, 1)

            # 4. Merge
            N, L, D = enc_out.shape
            merged_seq = enc_out.reshape(1, N * L, D)
            
            # 5. Global Position
            B, T, _ = merged_seq.shape
            limit = min(T, self.model.global_pos.size(0))
            pos_emb = self.model.global_pos[:limit, :].unsqueeze(0)

            if T > self.model.global_pos.size(0):
                 memory = merged_seq[:, :limit, :] + pos_emb
            else:
                 memory = merged_seq + pos_emb

            # ==========================================
            # 6. AUTO-DETECT BiLSTM (Universal Fix)
            # ==========================================
            # Checks if the model has 'context_bilstm' and runs it if it does
            if hasattr(self.model, 'context_bilstm'):
                self.model.context_bilstm.flatten_parameters()
                memory, _ = self.model.context_bilstm(memory)

            return memory

    def preprocess(self, image_input):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            image = Image.open(image_input).convert('L')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('L')
        else:
            raise ValueError("Input must be a file path or PIL Image.")

        target_height = 48
        aspect_ratio = image.width / image.height
        new_width = int(target_height * aspect_ratio)
        new_width = max(10, new_width) 
        
        image = image.resize((new_width, target_height), Image.Resampling.BILINEAR)
        img_tensor = self.transform(image) 
        
        # Helper function usage (ensure chunk_image_inference is defined)
        chunks = chunk_image_inference(img_tensor, chunk_width=100, overlap=16)
        chunks_norm = [(c - 0.5) / 0.5 for c in chunks]
        
        return torch.stack(chunks_norm).to(self.device)

    def encode_image(self, chunks_tensor):
        with torch.no_grad():
            # 1. CNN
            f = self.model.cnn(chunks_tensor)
            
            # 2. Patch
            patch_out = self.model.patch(f)
            p = patch_out[0] if isinstance(patch_out, tuple) else patch_out

            # 3. Encoder
            p = p.transpose(0, 1).contiguous()
            enc_out = self.model.enc(p)
            enc_out = enc_out.transpose(0, 1)

            # 4. Merge
            N, L, D = enc_out.shape
            merged_seq = enc_out.reshape(1, N * L, D)
            
            # 5. Global Position
            B, T, _ = merged_seq.shape
            limit = min(T, self.model.global_pos.size(0))
            pos_emb = self.model.global_pos[:limit, :].unsqueeze(0)

            if T > self.model.global_pos.size(0):
                 memory = merged_seq[:, :limit, :] + pos_emb
            else:
                 memory = merged_seq + pos_emb

            # ==========================================
            # 6. AUTO-DETECT BiLSTM (Universal Fix)
            # ==========================================
            # Checks if the model has 'context_bilstm' and runs it if it does
            if hasattr(self.model, 'context_bilstm'):
                self.model.context_bilstm.flatten_parameters()
                memory, _ = self.model.context_bilstm(memory)

            return memory

    def predict(self, image_input, max_len=128, beam_width=3):
        chunks_tensor = self.preprocess(image_input)
        memory = self.encode_image(chunks_tensor)

        if beam_width <= 1:
            tokens = self._greedy_decode(memory, max_len)
        else:
            tokens = self._beam_search(memory, max_len, beam_width)

        result_text = ""
        for idx in tokens:
            if idx == self.sos_idx or idx == self.pad_idx: continue
            if idx == self.eos_idx: break
            char = self.idx2char.get(idx, "")
            result_text += char

        return result_text

    def _greedy_decode(self, memory, max_len):
        B, T, _ = memory.shape
        memory_mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        generated = [self.sos_idx]

        with torch.no_grad():
            for _ in range(max_len):
                tgt = torch.LongTensor([generated]).to(self.device)
                logits = self.model.dec(tgt, memory, memory_mask)
                next_token = torch.argmax(logits[0, -1, :]).item()
                if next_token == self.eos_idx: break
                generated.append(next_token)
        return generated

    def _beam_search(self, memory, max_len, beam_width):
        B, T, D = memory.shape
        memory = memory.expand(beam_width, -1, -1)
        memory_mask = torch.zeros((beam_width, T), dtype=torch.bool, device=self.device)
        beams = [(0.0, [self.sos_idx])]
        completed_beams = []

        with torch.no_grad():
            for step in range(max_len):
                k_curr = len(beams)
                current_seqs = [b[1] for b in beams]
                tgt = torch.tensor(current_seqs, dtype=torch.long, device=self.device)
                step_logits = self.model.dec(tgt, memory[:k_curr], memory_mask[:k_curr])
                last_token_logits = step_logits[:, -1, :]
                log_probs = F.log_softmax(last_token_logits, dim=-1)

                candidates = []
                for i in range(k_curr):
                    score_so_far = beams[i][0]
                    seq_so_far = beams[i][1]
                    topk_probs, topk_idx = log_probs[i].topk(beam_width)
                    for k in range(beam_width):
                        token = topk_idx[k].item()
                        prob = topk_probs[k].item()
                        new_score = score_so_far + prob
                        new_seq = seq_so_far + [token]
                        candidates.append((new_score, new_seq))

                candidates.sort(key=lambda x: x[0], reverse=True)
                next_beams = []
                for score, seq in candidates:
                    if seq[-1] == self.eos_idx:
                        norm_score = score / (len(seq) - 1)
                        completed_beams.append((norm_score, seq))
                    else:
                        next_beams.append((score, seq))
                        if len(next_beams) == beam_width: break
                beams = next_beams
                if not beams: break

        if completed_beams:
            completed_beams.sort(key=lambda x: x[0], reverse=True)
            return completed_beams[0][1]
        elif beams:
            return beams[0][1]
        else:
            return [self.sos_idx]

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_PATH = "./test_img/2.png" 
    
    # Paths to model
    MODEL_PATH = "./checkpoints/khmerocr_vgg_lstm_epoch100.pth"
    CHAR2IDX_PATH = "char2idx.json"
    
    # Define where to save result
    # It will save as "textline_crop.txt"
    OUTPUT_TXT_PATH = os.path.splitext(IMAGE_PATH)[0] + ".txt"

    try:
        # 1. Initialize Inference Engine
        print("🚀 Initializing OCR Engine...")
        ocr_engine = KhmerOCRInference(MODEL_PATH, CHAR2IDX_PATH, model_class=KhmerOCR, emb_dim=384)

        # 2. Run Prediction
        print(f"📷 Processing: {IMAGE_PATH}")
        text = ocr_engine.predict(IMAGE_PATH, beam_width=3, max_len=256)

        # 3. Output to Console
        print("\n" + "="*30)
        print(f"RESULT: {text}")
        print("="*30)

        # 4. Save to File
        with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"\n💾 Saved result to: {OUTPUT_TXT_PATH}")

    except FileNotFoundError as e:
        print(f"❌ File Not Found: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()