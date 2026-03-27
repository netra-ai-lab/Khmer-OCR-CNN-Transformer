# ocr_engine.py
"""
Khmer OCR Pipeline — layout-aware, multi-format output.

Detection strategy
------------------
surya engine (default):
    Step 1a  LayoutPredictor   → labelled page regions
    Step 1b  DetectionPredictor → text lines *within* each text region
    Step 2   recognize_batch    → OCR text for each line crop
    Visual regions (Table, Picture, Formula) → cropped directly, no OCR

custom engine:
    Uses your trained LayoutInference detector.
    Same split: TEXT_IDS → OCR, VISUAL_IDS → crop.

Output formats (auto-detected from extension)
---------------------------------------------
  .txt  .md                    — OCR text only
  .html .pdf .docx             — every element placed at its exact bbox
                                 using the original pixel crop (no font rendering)
                                 → no black squares, no broken Khmer glyphs
"""

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
from .textline_detection import run_layout_and_lines, TEXT_LABELS, VISUAL_LABELS
from output_formatters import save_output, SUPPORTED_FORMATS

# Custom-model label IDs
_CUSTOM_TEXT_IDS   = {1, 2, 4, 5, 6, 8, 10, 11}
_CUSTOM_VISUAL_IDS = {3, 7, 9}
_CUSTOM_ID2LABEL   = {
    0: "Background", 1: "Caption",      2: "Footnote",
    3: "Formula",    4: "List-item",    5: "Page-footer",
    6: "Page-header",7: "Picture",      8: "Section-header",
    9: "Table",     10: "Text",         11: "Title",
}


class KhmerOCRPipeline:
    def __init__(self, engine: str = "surya"):
        """
        Args:
            engine: 'surya'  — Surya LayoutPredictor + DetectionPredictor (default)
                    'custom' — your trained LayoutInference model
        """
        self.engine = engine
        if engine == "custom":
            print("Initializing Custom Layout Detector...")
            self.detector = LayoutInference()
        else:
            print("Initializing Surya Pipeline (layout + text-line detection)...")

    # ------------------------------------------------------------------
    def process_image(
        self,
        image_path: str,
        output_path: str = None,
        save_debug: bool = False,
        padding: int = 8,
        beam_width: int = 1,
        batch_size: int = 8,
    ) -> str:
        """
        Run the full OCR pipeline.

        Each element in the page becomes one of:
          • text segment  — has "text" (OCR result) + "crop" (pixel image)
          • image segment — has "crop" only  (Table / Picture / Formula)

        Layout-preserving outputs (html/pdf/docx) use the crop images so
        the result is visually identical to the source regardless of font
        availability.

        Returns:
            Plain newline-joined OCR text (backward-compatible).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        image_size = img.size   # (width, height)

        # ----------------------------------------------------------------
        # STEP 1 — DETECTION
        # ----------------------------------------------------------------
        print(f"\nStep 1: Layout + line detection [{self.engine}] ...")

        if self.engine == "surya":
            regions = run_layout_and_lines(
                image_path,
                expansion_px=2,
                padding_px=padding,
            )
        else:
            regions = self._custom_detect(img, image_path, padding)

        if not regions:
            print("No elements detected.")
            return ""

        # ----------------------------------------------------------------
        # STEP 2 — COLLECT CROPS FOR OCR
        # Only text-type regions have lines that need to be recognised.
        # Visual regions already have their final crop stored.
        # ----------------------------------------------------------------

        # Flat list of (line_crop, line_bbox, region_label, region_bbox)
        # for ALL text lines across all text regions.
        ocr_queue = []
        for region in regions:
            if region["label"] in VISUAL_LABELS:
                continue  # no OCR needed
            for line in region["lines"]:
                ocr_queue.append((
                    line["crop"],
                    line["bbox"],
                    region["label"],
                    region["bbox"],
                ))

        print(f"\nStep 2: Recognising {len(ocr_queue)} text lines "
              f"(batch={batch_size}, beam={beam_width}) ...")

        if ocr_queue:
            line_crops = [q[0] for q in ocr_queue]
            recognitions = recognize_batch(
                line_crops, beam_width=beam_width, batch_size=batch_size
            )
        else:
            recognitions = []

        # ----------------------------------------------------------------
        # STEP 3 — BUILD FLAT SEGMENT LIST
        # Merge OCR results back and flatten everything into one list,
        # sorted by vertical position (Y of bbox top).
        # ----------------------------------------------------------------
        segments: list[dict] = []
        ocr_iter = iter(recognitions)

        for region in regions:
            label = region["label"]

            if label in VISUAL_LABELS:
                # One image segment per visual region
                assert len(region["lines"]) == 1, "Visual region should have exactly one line"
                line = region["lines"][0]
                segments.append({
                    "type":  "image",
                    "crop":  line["crop"],
                    "bbox":  line["bbox"],
                    "label": label,
                })

            else:
                # One text segment per detected line inside this region
                for line in region["lines"]:
                    text = next(ocr_iter, "")
                    segments.append({
                        "type":  "text",
                        "text":  text,
                        "crop":  line["crop"],   # kept for layout-preserving formats
                        "bbox":  line["bbox"],
                        "label": label,
                    })

        # Final sort: top-to-bottom, left-to-right
        segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))

        # ----------------------------------------------------------------
        # STEP 4 — SAVE
        # ----------------------------------------------------------------
        if save_debug:
            self._save_debug(image_path, segments)

        final_text = "\n".join(
            s["text"] for s in segments
            if s["type"] == "text" and s["text"].strip()
        )

        if output_path:
            print(f"\nStep 3: Saving → {output_path}")
            save_output(segments, output_path, image_size=image_size, image_path=image_path)

        return final_text

    # ------------------------------------------------------------------
    def _custom_detect(self, img: Image.Image, image_path: str, padding: int) -> list:
        """
        Wrap the custom LayoutInference model output into the same
        region-dict format that run_layout_and_lines() returns.
        """
        from .textline_detection import _crop_padded, _extract_lines_in_region
        from surya.detection import DetectionPredictor

        results      = self.detector.run(image_path)
        raw_elements = (
            results.get("elements", []) if isinstance(results, dict) else results
        )
        raw_elements = sorted(raw_elements, key=lambda e: e[0][1])

        iw, ih = img.size
        det_predictor = DetectionPredictor()
        regions = []

        for el in raw_elements:
            x1, y1, x2, y2 = el[0]
            label_id = el[1] if len(el) > 1 else 10
            label    = _CUSTOM_ID2LABEL.get(label_id, "Text")

            if label_id == 0:
                continue

            x1=max(0,x1); y1=max(0,y1); x2=min(iw,x2); y2=min(ih,y2)
            if x2-x1 < 4 or y2-y1 < 4:
                continue

            if label_id in _CUSTOM_VISUAL_IDS:
                crop  = _crop_padded(img, x1, y1, x2, y2, pad=0)
                lines = [{"bbox": (x1, y1, x2, y2), "crop": crop}]
            else:
                lines = _extract_lines_in_region(
                    img, (x1, y1, x2, y2), det_predictor,
                    expansion_px=2, padding_px=padding,
                )

            regions.append({"label": label, "bbox": (x1,y1,x2,y2), "lines": lines})

        return regions

    # ------------------------------------------------------------------
    def _save_debug(self, image_path: str, segments: list) -> None:
        base      = os.path.splitext(os.path.basename(image_path))[0]
        folder    = f"debug_{base}_{self.engine}"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        txt_i = img_i = 0
        for seg in segments:
            if seg["type"] == "text":
                seg["crop"].save(os.path.join(folder, f"text_{txt_i:03d}.png"))
                with open(os.path.join(folder, f"text_{txt_i:03d}.txt"),
                          "w", encoding="utf-8") as fh:
                    fh.write(seg["text"])
                txt_i += 1
            else:
                seg["crop"].save(os.path.join(folder, f"{seg['label'].lower()}_{img_i:03d}.png"))
                img_i += 1

        print(f"  [Debug] {txt_i} text + {img_i} image crops saved to '{folder}/'")


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Khmer OCR — layout-aware pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Supported output formats: {', '.join(SUPPORTED_FORMATS)}\n\n"
            "Examples:\n"
            "  python ocr_engine.py --image scan.jpg --output result.html\n"
            "  python ocr_engine.py --image scan.jpg --output result.docx\n"
            "  python ocr_engine.py --image scan.jpg --output result.pdf  --debug\n"
            "  python ocr_engine.py --image scan.jpg --output result.txt  --engine custom\n"
        ),
    )
    parser.add_argument("--image",      required=True)
    parser.add_argument("--engine",     choices=["custom", "surya"], default="surya")
    parser.add_argument("--output",     default="ocr_result.txt",
                        help="Extension determines format: .txt .md .html .pdf .docx")
    parser.add_argument("--padding",    type=int, default=8)
    parser.add_argument("--beam",       type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    try:
        pipeline = KhmerOCRPipeline(engine=args.engine)
        pipeline.process_image(
            image_path  = args.image,
            output_path = args.output,
            save_debug  = args.debug,
            padding     = args.padding,
            beam_width  = args.beam,
            batch_size  = args.batch_size,
        )
    except Exception as exc:
        print(f"\nPipeline error: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()