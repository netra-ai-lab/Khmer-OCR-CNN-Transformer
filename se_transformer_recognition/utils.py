import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

def autodetect_config(model_path: str | Path) -> dict:
    """
    Peeks into the .pth checkpoint to infer model dimensions.
    Returns a dictionary of overrides for OCRConfig.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    logger.info(f"Inspecting checkpoint: {path.name}...")
    # Load on CPU just to check shapes (fast)
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    detected = {}

    # 1. Detect Embedding Dim & Encoder Seq Len from 'global_pos'
    if 'global_pos' in state_dict:
        shape = state_dict['global_pos'].shape
        detected['max_seq_len'] = shape[0]
        detected['emb_dim'] = shape[1]
        logger.info(f"   ↳ Auto-detected: emb_dim={shape[1]}, max_seq_len={shape[0]}")
    
    # 2. Detect Decoder Max Length
    if 'dec.pos_emb' in state_dict:
        shape = state_dict['dec.pos_emb'].shape
        detected['decode_max_len'] = shape[0]
        logger.info(f"   ↳ Auto-detected: decode_max_len={shape[0]}")

    return detected