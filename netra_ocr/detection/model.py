# model.py
import torch
from transformers import SegformerForSemanticSegmentation
from .config import Config

class LayoutModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading weights from {Config.MODEL_PATH}...")
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            Config.MODEL_PATH, 
            num_labels=Config.NUM_LABELS,
            id2label=Config.ID2LABEL,
            label2id={v: k for k, v in Config.ID2LABEL.items()},
            local_files_only=True
        ).to(self.device)
        self.model.eval()

    def predict(self, pixel_values, original_size):
        with torch.no_grad():
            outputs = self.model(pixel_values.to(self.device))
            logits = outputs.logits

        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=original_size[::-1], mode="bilinear", align_corners=False
        )
        
        # Get Class Map and Heatmap
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        probs = torch.nn.functional.softmax(upsampled_logits, dim=1)[0]
        pred_heatmap = torch.max(probs, dim=0)[0].cpu().numpy()
        
        return pred_seg, pred_heatmap