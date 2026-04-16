import json
from pathlib import Path

class Tokenizer:
    """Handles mapping between characters and integer IDs."""
    
    def __init__(self, char2idx_path: str | Path):
        self.char2idx_path = Path(char2idx_path)
        self.char2idx, self.idx2char = self._load_vocab()
        
        # Special Tokens (ensure these match your JSON keys)
        self.sos_idx = self.char2idx.get("<sos>", 1)
        self.eos_idx = self.char2idx.get("<eos>", 2)
        self.pad_idx = self.char2idx.get("<pad>", 0)

    def _load_vocab(self) -> tuple[dict, dict]:
        if not self.char2idx_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {self.char2idx_path}")
        
        with open(self.char2idx_path, 'r', encoding='utf-8') as f:
            char2idx = json.load(f)
        
        idx2char = {v: k for k, v in char2idx.items()}
        return char2idx, idx2char

    def decode(self, token_ids: list[int]) -> str:
        """Converts a list of token IDs back to a string."""
        result = []
        for idx in token_ids:
            if idx == self.sos_idx or idx == self.pad_idx:
                continue
            if idx == self.eos_idx:
                break
            result.append(self.idx2char.get(idx, ""))
        return "".join(result)

    def __len__(self):
        return len(self.char2idx)