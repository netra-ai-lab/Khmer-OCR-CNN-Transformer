import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from PIL import Image

from .config import Config
from .preprocessor import Preprocessor
from .model import LayoutModel
from .utils import extract_layout_elements

class LayoutInference:
    def __init__(self):
        print("Initializing Layout Detection Engine...")
        self.prep = Preprocessor()
        self.model = LayoutModel()

    def run(self, image_path, output_dir=None, show=False):
        """
        Main execution logic for a single image.
        """
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return

        print(f"Processing: {image_path}")
        image, pixel_values, orig_size = self.prep.prepare_image(image_path)
        pred_seg, pred_heatmap = self.model.predict(pixel_values, orig_size)

        print("Running layout extraction...")
        # Note: extract_layout_elements returns (crops, boxes)
        _, refined_results = extract_layout_elements(image, pred_seg, pred_heatmap)

        # 3. Save results (Using the old pred_heatmap glow)
        target_dir = output_dir if output_dir else Config.OUTPUT_DIR
        self.save_results(image_path, image, pred_heatmap, refined_results, target_dir)

        # 4. Optional Visual Show
        if show:
            self.visualize_inline(image, refined_results)
            
        return refined_results

    def save_results(self, image_path, original_img, pred_heatmap, refined_results, base_output_dir):
        """
        Saves the colorful Heatmap, Bounding Box Visualization, and JSON data.
        """
        base_name = os.path.basename(image_path).split('.')[0]
        task_dir = os.path.join(base_output_dir, base_name)
        os.makedirs(task_dir, exist_ok=True)

        # A. Save Heatmap
        heatmap_8bit = (pred_heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(task_dir, f"{base_name}_heatmap.png"), heatmap_color)

        # B. Prepare Box Viz & JSON Data
        viz_img = np.array(original_img.copy())
        viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR) 
        
        json_output = []
        for (coords, cls_id) in refined_results:
            x1, y1, x2, y2 = map(int, coords)
            label_name = Config.ID2LABEL.get(cls_id, "Unknown")
            color = Config.COLORS.get(cls_id, (255, 255, 255))
            cv_color = (color[2], color[1], color[0]) # RGB -> BGR for OpenCV

            cv2.rectangle(viz_img, (x1, y1), (x2, y2), cv_color, Config.DRAW_THICKNESS)
            
            json_output.append({
                "category": label_name,
                "category_id": int(cls_id),
                "bbox": [x1, y1, x2, y2]
            })

        # Save Visual Check
        cv2.imwrite(os.path.join(task_dir, f"{base_name}_boxes.png"), viz_img)

        # C. Save JSON (Reading Order)
        json_output.sort(key=lambda x: x["bbox"][1])
        output_data = {
            "filename": os.path.basename(image_path),
            "dimensions": {"width": original_img.width, "height": original_img.height},
            "elements": json_output
        }

        with open(os.path.join(task_dir, f"{base_name}_layout.json"), "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)

        print(f"Results saved to: {task_dir}")

    def visualize_inline(self, image, refined_results):
        viz_img_rgb = np.array(image.copy())
        for (coords, cls_id) in refined_results:
            x1, y1, x2, y2 = map(int, coords)
            color = Config.COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(viz_img_rgb, (x1, y1), (x2, y2), color, 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(viz_img_rgb)
        plt.title(f"Detected {len(refined_results)} elements")
        plt.axis("off")
        plt.show()

# ==========================================
# CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer Layout Detection CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Custom output directory")
    parser.add_argument("--show", action="store_true", help="Display plot")
    
    args = parser.parse_args()

    engine = LayoutInference()
    engine.run(args.image, output_dir=args.output, show=args.show)

    """
    USAGE EXAMPLE:

        python detection.py --image goc_3.tiff --show --output results/custom_dir/
        
    ARGUMENTS:
        --image: Path to the input image (Required).
        --show: Flag to display visualization inline (Optional).
        --output: Custom output directory (Optional).
    
    RUN VIA PYTHON:
        from detection import LayoutInference

        engine = LayoutInference()
        # This will return the list of bounding boxes
        results = engine.run("document.png") 
    """