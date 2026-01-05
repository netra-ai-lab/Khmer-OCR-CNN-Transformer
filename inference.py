import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
from vgg_model import KhmerOCR 
from textline_detection import run_textline_detector

# ==============================================================================
# 2. HELPER FUNCTIONS & INFERENCE CLASS
# ==============================================================================

def chunk_image_inference(img_tensor, chunk_width=100, overlap=16):
    """
    Splits image into chunks. Matches training logic.
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
            # F.pad: (left, right, top, bottom)
            chunk = F.pad(chunk, (0, pad_size, 0, 0), value=1.0)

        chunks.append(chunk)
        start += chunk_width - overlap

    return chunks

class KhmerOCRInference:
    def __init__(self, model_path, char2idx_input, device='cuda'):
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
        self.model = KhmerOCR(
            vocab_size=len(self.char2idx),
            pad_idx=self.pad_idx,
            emb_dim=384,
            max_global_len=4096
        )

        # 3. Load Weights
        print(f"⏳ Loading recognition weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()

        # 4. Transform
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(), # 0..1
        ])

    def preprocess(self, image_input):
        # Handle Input Type
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('L')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('L')
        else:
            raise ValueError("Input must be a file path or PIL Image.")

        # --- RESIZE TO HEIGHT 48 ---
        target_height = 48
        aspect_ratio = image.width / image.height
        new_width = int(target_height * aspect_ratio)
        image = image.resize((new_width, target_height), Image.Resampling.BILINEAR)
        # ---------------------------

        img_tensor = self.transform(image) 
        chunks = chunk_image_inference(img_tensor, chunk_width=100, overlap=16)
        chunks_norm = [(c - 0.5) / 0.5 for c in chunks]
        return torch.stack(chunks_norm).to(self.device)

    def encode_image(self, chunks_tensor):
        with torch.no_grad():
            f = self.model.cnn(chunks_tensor)
            p, _ = self.model.patch(f)
            p = p.transpose(0, 1).contiguous()
            enc_out = self.model.enc(p)
            enc_out = enc_out.transpose(0, 1)

            N, L, D = enc_out.shape
            merged_seq = enc_out.reshape(1, N * L, D)
            memory = merged_seq

            B, T, _ = memory.shape
            limit = min(T, self.model.global_pos.size(0))
            pos_emb = self.model.global_pos[:limit, :].unsqueeze(0)

            if T > self.model.global_pos.size(0):
                 memory = memory[:, :limit, :] + pos_emb
            else:
                 memory = memory + pos_emb

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
# 3. FULL PIPELINE (Surya Detect -> Custom Recognize)
# ==============================================================================

def run_full_document_ocr(image_path, model_path, vocab_input):
    
    # 1. Initialize Recognition Model ONCE
    ocr_model = KhmerOCRInference(model_path, vocab_input)
    
    print(f"🔍 Running detection on: {image_path}")
    
    # 2. Run Detection (Using your imported function)
    textline_results = run_textline_detector(
        image_path, 
        expansion_px=5,  # Expands the box 5px into the image
        padding_px=10    # Adds 10px white border
    )
    textlines, _ = textline_results

    if len(textlines) == 0:
        print("⚠️ No textlines detected.")
        return []

    print(f"✅ Detected {len(textlines)} text lines. Starting recognition...")

    # 3. Sort lines by vertical center
    def line_y_center(item):
        _, (xmin, ymin, xmax, ymax) = item
        return (ymin + ymax) / 2

    textlines = sorted(textlines, key=line_y_center)

    results = []

    # 4. Recognize each line
    for i, (crop_img, bbox) in enumerate(textlines):
        text = ocr_model.predict(crop_img, beam_width=3)
        results.append({
            "line_number": i,
            "text": text,
            "bbox": bbox
        })
        print(f"   Line {i}: {text}")

    return results

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_PATH = "test_image.png"
    MODEL_PATH = "./checkpoints/khmerocr_epoch100.pth"
    CHAR2IDX_PATH = "char2idx.json"
    
    # Define Output Folder
    RESULT_FOLDER = "results"

    try:
        # 1. Setup Output Directory
        if not os.path.exists(RESULT_FOLDER):
            os.makedirs(RESULT_FOLDER)
            print(f"📁 Created folder: {RESULT_FOLDER}")

        # 2. Determine Output Filename (Same as image name but .txt)
        base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        output_txt_path = os.path.join(RESULT_FOLDER, f"{base_name}.txt")

        # 3. Load Vocab
        with open(CHAR2IDX_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # 4. Run Pipeline
        final_output = run_full_document_ocr(IMAGE_PATH, MODEL_PATH, vocab)

        # 5. Save Results to .txt
        print(f"\n💾 Saving results to: {output_txt_path}")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for line in final_output:
                f.write(line['text'] + "\n")

        print("✅ Success! Process completed.")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()