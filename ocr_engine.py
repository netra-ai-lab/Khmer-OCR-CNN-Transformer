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
from textline_detection import run_textline_detector

class KhmerOCRPipeline:
    def __init__(self, engine="surya"):
        """
        Initializes the full OCR pipeline.
        
        Args:
            engine (str): 'custom' for your trained model, 'surya' for Surya detector.
        """
        self.engine = engine
        
        self.ID2LABEL = {
            0: "Background", 1: "Caption", 2: "Footnote", 3: "Formula", 
            4: "List-item", 5: "Page-footer", 6: "Page-header", 7: "Picture", 
            8: "Section-header", 9: "Table", 10: "Text", 11: "Title"
        }
        self.TEXT_IDS = [1, 2, 4, 5, 6, 8, 10, 11]

        if self.engine == "custom":
            print("Initializing Custom Layout Detector...")
            self.detector = LayoutInference()
        else:
            print("Initializing Surya Layout Detector...")
            pass

    def process_image(self, 
                      image_path: str, 
                      output_path: str = None, 
                      save_debug: bool = False, 
                      padding: int = 6,
                      beam_width: int = 1,
                      batch_size: int = 8):
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img = Image.open(image_path).convert("RGB")
        crops = []

        # ---------------------------------------------------------
        # STEP 1: DETECTION
        # ---------------------------------------------------------
        if self.engine == "surya":
            print(f"Step 1: Detecting layout using SURYA for {image_path}...")
            crops_with_coords, _ = run_textline_detector(
                image_path, 
                expansion_px=2, 
                padding_px=padding
            )
            crops = [c[0] for c in crops_with_coords]
        
        else:
            print(f"Step 1: Detecting layout using CUSTOM model for {image_path}...")
            results = self.detector.run(image_path)
            raw_elements = results.get("elements", []) if isinstance(results, dict) else results
            
            valid_elements = [e for e in raw_elements if e[1] in self.TEXT_IDS]
            valid_elements.sort(key=lambda x: x[0][1]) # Sort by Y

            img_w, img_h = img.size
            for el in valid_elements:
                x1, y1, x2, y2 = el[0]
                px1 = max(0, x1 - padding); py1 = max(0, y1 - padding)
                px2 = min(img_w, x2 + padding); py2 = min(img_h, y2 + padding)
                crops.append(img.crop((px1, py1, px2, py2)))

        if not crops:
            print("No text elements found.")
            return ""

        print(f"Step 2: Processing {len(crops)} lines (Batch Size: {batch_size}, Engine: {self.engine})...")
        recognitions = recognize_batch(crops, beam_width=beam_width, batch_size=batch_size)
        
        if save_debug:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            debug_folder = f"debug_{base_name}_{self.engine}"
            if os.path.exists(debug_folder): shutil.rmtree(debug_folder)
            os.makedirs(debug_folder, exist_ok=True)
            
            for i, (crop, text) in enumerate(zip(crops, recognitions)):
                crop.save(os.path.join(debug_folder, f"line_{i:03d}.png"))
                with open(os.path.join(debug_folder, f"line_{i:03d}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)

        final_text = "\n".join(recognitions)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(f"Step 3: Full text saved to {output_path}")
        
        return final_text

def main():
    parser = argparse.ArgumentParser(description="Khmer OCR Holistic Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--engine", type=str, choices=["custom", "surya"], default="surya", 
                        help="Which detection engine to use (default: surya)")
    parser.add_argument("--output", type=str, default="ocr_result.txt", help="Path to save result")
    parser.add_argument("--padding", type=int, default=6, help="Padding/White-space around lines")
    parser.add_argument("--beam", type=int, default=1, help="Beam width for OCR")
    parser.add_argument("--batch_size", type=int, default=8, help="Recognition batch size")
    parser.add_argument("--debug", action="store_true", help="Save debug crops")

    args = parser.parse_args()

    try:
        pipeline = KhmerOCRPipeline(engine=args.engine)
        pipeline.process_image(
            image_path=args.image,
            output_path=args.output,
            save_debug=args.debug,
            padding=args.padding,
            beam_width=args.beam,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"\nPipeline Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()