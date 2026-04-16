from dataclasses import dataclass
import torch

@dataclass
class OCRConfig:
    """Configuration for OCR Inference Pipeline."""
    img_height: int = 48
    chunk_width: int = 100
    chunk_overlap: int = 16
    emb_dim: int = 384        
    max_seq_len: int = 4096   
    decode_max_len: int = 256 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"