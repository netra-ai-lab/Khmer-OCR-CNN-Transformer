import os
import shutil
import argparse
import sys
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from detection.detector import LayoutInference
from recognition.recognize_text import recognize_batch

class KhmerOCRPipeline:
    def __init__(self):
        """
        Initializes the full OCR pipeline. 
        Loads detection and recognition models into memory.
        """
        # DocLayNet Label Mapping
        self.ID2LABEL = {
            0: "Background", 1: "Caption", 2: "Footnote", 3: "Formula", 
            4: "List-item", 5: "Page-footer", 6: "Page-header", 7: "Picture", 
            8: "Section-header", 9: "Table", 10: "Text", 11: "Title"
        }
        
        # Define IDs we want to process (Textual elements)
        self.TEXT_IDS = [1, 2, 4, 5, 6, 8, 10, 11]
        
        print("Initializing Layout Detector...")
        self.detector = LayoutInference()

    def process_image(self, 
                      image_path: str, 
                      output_path: str = None, 
                      save_debug: bool = False, 
                      padding: int = 10,
                      beam_width: int = 1,
                      batch_size: int = 8):
        """
        Performs full OCR: Detection -> Cropping -> Recognition.
        
        Returns:
            str: The full recognized text.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 1. Detection
        print(f"Step 1: Detecting layout for {image_path}...")
        results = self.detector.run(image_path)
        raw_elements = results.get("elements", []) if isinstance(results, dict) else results
        
        # 2. Filtering & Sorting
        valid_elements = []
        for raw_e in raw_elements:
            if raw_e[1] in self.TEXT_IDS:
                valid_elements.append({"bbox": raw_e[0]})

        # Sort by Y coordinate (Top-to-Bottom reading order)
        valid_elements.sort(key=lambda x: x['bbox'][1])

        if not valid_elements:
            print("No text elements found.")
            return ""

        # 3. CROP IN MEMORY with Padding
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size
        
        crops = []
        for el in valid_elements:
            x1, y1, x2, y2 = el['bbox']
            
            # Apply padding while staying within image boundaries
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(img_w, x2 + padding)
            py2 = min(img_h, y2 + padding)
            
            crops.append(img.crop((px1, py1, px2, py2)))

        # 4. BATCH RECOGNITION
        print(f"Step 2: Processing {len(crops)} text lines (Batch Size: {batch_size})...")
        recognitions = recognize_batch(crops, beam_width=beam_width, batch_size=batch_size)

        # 5. DEBUG: Save individual crops and their text
        if save_debug:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            debug_folder = f"debug_{base_name}"
            
            if os.path.exists(debug_folder):
                shutil.rmtree(debug_folder)
            os.makedirs(debug_folder, exist_ok=True)
            
            print(f"Saving debug crops to: {debug_folder}")
            for i, (crop, text) in enumerate(zip(crops, recognitions)):
                crop_filename = os.path.join(debug_folder, f"line_{i:03d}.png")
                crop.save(crop_filename)
                with open(os.path.join(debug_folder, f"line_{i:03d}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)

        # 6. Final Output
        final_text = "\n".join(recognitions)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(f"Step 3: Full text saved to {output_path}")
        
        return final_text

def main():
    parser = argparse.ArgumentParser(description="Khmer OCR Holistic Pipeline")
    
    # Required Arguments
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    
    # Optional Arguments
    parser.add_argument("--output", type=str, default="ocr_result.txt", help="Path to save recognized text")
    parser.add_argument("--padding", type=int, default=6, help="Padding around crops (default: 5)")
    parser.add_argument("--beam", type=int, default=1, help="Beam width for recognition (1=Greedy, faster)")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of lines to process at once")
    parser.add_argument("--debug", action="store_true", help="Save cropped images for debugging")

    args = parser.parse_args()

    try:
        pipeline = KhmerOCRPipeline()
        pipeline.process_image(
            image_path=args.image,
            output_path=args.output,
            save_debug=args.debug,
            padding=args.padding,
            beam_width=args.beam,
            batch_size=args.batch_size
        )
        print("\nPipeline execution finished successfully.")
    except Exception as e:
        print(f"\nPipeline Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()