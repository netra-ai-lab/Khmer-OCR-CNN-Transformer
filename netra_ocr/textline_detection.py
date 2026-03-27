# textline_detection.py
"""
Surya-based detection helpers.

Two predictors are used:
  LayoutPredictor  — classifies the page into labelled regions
                     (Title, Text, Table, Picture, Formula, …)
  DetectionPredictor — finds individual text lines within a region

Public API
----------
run_layout_and_lines(image_path, expansion_px, padding_px)
    → List[dict]   (one dict per page element, sorted top-to-bottom)

Each dict:
    {
        "label":  str,            # "Title" | "Text" | "Table" | "Picture" | …
        "bbox":   (x1,y1,x2,y2), # in original image pixel coordinates
        "lines":  [ {"bbox": (x1,y1,x2,y2), "crop": PIL.Image}, … ]
                  # text-type regions: one entry per detected text line
                  # visual regions  : single entry = the region crop itself
    }
"""

from __future__ import annotations
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
# Regions where we want to run text-line detection
TEXT_LABELS = {
    "Title", "Caption", "Footnote", "Formula",
    "List-item", "Page-footer", "Page-header",
    "Section-header", "Text",
}
# Regions that are captured as-is (no OCR)
VISUAL_LABELS = {"Picture", "Figure", "Table"}

# All known Surya layout labels
ALL_LABELS = TEXT_LABELS | VISUAL_LABELS


# ---------------------------------------------------------------------------
# Crop + pad helper
# ---------------------------------------------------------------------------
def _crop_padded(image: Image.Image, x1: int, y1: int, x2: int, y2: int,
                 pad: int = 0) -> Image.Image:
    iw, ih = image.size
    px1 = max(0, x1 - pad);  py1 = max(0, y1 - pad)
    px2 = min(iw, x2 + pad); py2 = min(ih, y2 + pad)
    crop = image.crop((px1, py1, px2, py2))
    if pad > 0:
        canvas = Image.new("RGB", (crop.width + pad * 2, crop.height + pad * 2), (255, 255, 255))
        canvas.paste(crop, (pad, pad))
        return canvas
    return crop


# ---------------------------------------------------------------------------
# Text-line extraction within one region
# ---------------------------------------------------------------------------
def _extract_lines_in_region(
    full_image: Image.Image,
    region_bbox: Tuple[int, int, int, int],
    det_predictor,
    expansion_px: int = 2,
    padding_px: int = 8,
) -> List[dict]:
    """
    Run text-line detection on a single layout region.
    Returns list of {"bbox": (x1,y1,x2,y2), "crop": PIL.Image}
    where bbox coords are in the FULL page coordinate space.
    """
    rx1, ry1, rx2, ry2 = region_bbox
    iw, ih = full_image.size

    # Crop the region (with small expansion so edge characters aren't clipped)
    region_crop = _crop_padded(full_image, rx1, ry1, rx2, ry2, pad=expansion_px)

    # Run text-line detector on the region crop
    pred = det_predictor([region_crop])[0]
    if not pred.bboxes:
        # Fallback: treat entire region as one line
        crop = _crop_padded(full_image, rx1, ry1, rx2, ry2, pad=padding_px)
        return [{"bbox": (rx1, ry1, rx2, ry2), "crop": crop}]

    lines = []
    for obj in pred.bboxes:
        poly  = obj.polygon
        xs    = [p[0] for p in poly]
        ys    = [p[1] for p in poly]
        lx0   = int(min(xs)); ly0 = int(min(ys))
        lx1   = int(max(xs)); ly1 = int(max(ys))

        # Expand within region crop
        lx0 = max(0, lx0 - expansion_px); ly0 = max(0, ly0 - expansion_px)
        lx1 = min(region_crop.width,  lx1 + expansion_px)
        ly1 = min(region_crop.height, ly1 + expansion_px)

        if lx1 - lx0 <= 0 or ly1 - ly0 <= 0:
            continue

        # Offset back to full-page coordinates
        # (region_crop has expansion_px baked in on each side, so subtract it)
        page_x1 = rx1 - expansion_px + lx0
        page_y1 = ry1 - expansion_px + ly0
        page_x2 = rx1 - expansion_px + lx1
        page_y2 = ry1 - expansion_px + ly1

        # Clamp to page
        page_x1 = max(0, page_x1); page_y1 = max(0, page_y1)
        page_x2 = min(iw, page_x2); page_y2 = min(ih, page_y2)

        # Crop with padding from the FULL source image (best quality)
        crop = _crop_padded(full_image, page_x1, page_y1, page_x2, page_y2,
                            pad=padding_px)
        lines.append({"bbox": (page_x1, page_y1, page_x2, page_y2), "crop": crop})

    # Sort text lines top → bottom
    lines.sort(key=lambda l: l["bbox"][1])
    return lines


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def run_layout_and_lines(
    image_path: str,
    expansion_px: int = 2,
    padding_px: int = 8,
) -> List[dict]:
    """
    Full layout-aware detection pipeline using Surya.

    1. LayoutPredictor  → labelled regions sorted top-to-bottom
    2. For TEXT_LABELS  → DetectionPredictor within each region
    3. For VISUAL_LABELS→ region cropped directly (no line detection)

    Returns a list of region dicts, each containing:
        label   : str
        bbox    : (x1, y1, x2, y2)  — page coordinates
        lines   : list of {"bbox": ..., "crop": PIL.Image}
    """
    from surya.detection  import DetectionPredictor
    from surya.layout     import LayoutPredictor
    from surya.foundation import FoundationPredictor
    from surya.settings   import settings

    image = Image.open(image_path).convert("RGB")
    iw, ih = image.size

    # ---- 1. Layout detection -----------------------------------------------
    # Surya 0.17+: LayoutPredictor requires a FoundationPredictor with the
    # layout model checkpoint.
    print("  [Detection] Running Surya LayoutPredictor ...")
    foundation_predictor = FoundationPredictor(
        checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
    )
    layout_predictor = LayoutPredictor(foundation_predictor)
    layout_result    = layout_predictor([image])[0]

    # layout_result.bboxes: list of LayoutBox
    # LayoutBox.bbox  : [x1, y1, x2, y2]
    # LayoutBox.label : str
    if not layout_result.bboxes:
        print("  [Detection] No layout regions found — falling back to full-page text-line detection")
        det_predictor = DetectionPredictor()
        return _fallback_full_page(image, det_predictor, expansion_px, padding_px)

    # Sort regions top-to-bottom (primary) then left-to-right (secondary)
    raw_regions = sorted(
        layout_result.bboxes,
        key=lambda b: (b.bbox[1], b.bbox[0]),
    )
    print(f"  [Detection] Found {len(raw_regions)} layout regions: "
          f"{[r.label for r in raw_regions]}")

    # ---- 2. Text-line detection in text regions ----------------------------
    det_predictor = DetectionPredictor()

    regions_out: List[dict] = []
    for lb in raw_regions:
        label = lb.label
        x1, y1, x2, y2 = [int(v) for v in lb.bbox]

        # Clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(iw, x2); y2 = min(ih, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        if label in VISUAL_LABELS:
            # Embed the entire region as one image crop
            crop  = _crop_padded(image, x1, y1, x2, y2, pad=0)
            lines = [{"bbox": (x1, y1, x2, y2), "crop": crop}]

        elif label in TEXT_LABELS:
            lines = _extract_lines_in_region(
                image, (x1, y1, x2, y2), det_predictor,
                expansion_px=expansion_px, padding_px=padding_px,
            )

        else:
            # Unknown label — treat as text
            lines = _extract_lines_in_region(
                image, (x1, y1, x2, y2), det_predictor,
                expansion_px=expansion_px, padding_px=padding_px,
            )

        regions_out.append({
            "label": label,
            "bbox":  (x1, y1, x2, y2),
            "lines": lines,
        })

    return regions_out


def _fallback_full_page(image, det_predictor, expansion_px, padding_px):
    """Used when LayoutPredictor finds nothing — whole page as one Text region."""
    pred = det_predictor([image])[0]
    lines = []
    iw, ih = image.size
    for obj in pred.bboxes:
        poly = obj.polygon
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x0,y0,x1,y1 = int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))
        x0=max(0,x0-expansion_px); y0=max(0,y0-expansion_px)
        x1=min(iw,x1+expansion_px); y1=min(ih,y1+expansion_px)
        if x1-x0>0 and y1-y0>0:
            crop = _crop_padded(image, x0, y0, x1, y1, pad=padding_px)
            lines.append({"bbox":(x0,y0,x1,y1),"crop":crop})
    lines.sort(key=lambda l: l["bbox"][1])
    return [{"label": "Text", "bbox": (0,0,iw,ih), "lines": lines}]


# ---------------------------------------------------------------------------
# Legacy function kept for backward compatibility
# ---------------------------------------------------------------------------
def run_textline_detector(image_path, expansion_px=5, padding_px=10):
    """
    Original text-line-only detector (no layout classification).
    Kept for backward compatibility.
    Returns (crops_sorted, raw_prediction).
    """
    from surya.detection import DetectionPredictor
    image = Image.open(image_path).convert("RGB")
    det_predictor = DetectionPredictor()
    textline_pred = det_predictor([image])[0]
    crops = extract_textline_crops(image, textline_pred, expansion_px, padding_px)
    crops_sorted = sorted(crops, key=lambda x: x[1][1])
    return crops_sorted, textline_pred


def extract_textline_crops(image, textline_pred, expansion_px=5, padding_px=10):
    crops = []
    iw, ih = image.size
    for obj in textline_pred.bboxes:
        poly = obj.polygon
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x0,y0 = int(min(xs)),int(min(ys))
        x1,y1 = int(max(xs)),int(max(ys))
        x0=max(0,x0-expansion_px); y0=max(0,y0-expansion_px)
        x1=min(iw,x1+expansion_px); y1=min(ih,y1+expansion_px)
        if x1-x0<=0 or y1-y0<=0: continue
        crop = image.crop((x0,y0,x1,y1))
        if padding_px > 0:
            padded = Image.new("RGB",(crop.width+padding_px*2,crop.height+padding_px*2),(255,255,255))
            padded.paste(crop,(padding_px,padding_px))
            crop = padded
        crops.append((crop,(x0,y0,x1,y1)))
    return crops


# ---------------------------------------------------------------------------
# Visualization utils (unchanged)
# ---------------------------------------------------------------------------
try:
    FONT = ImageFont.truetype("arial.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()

def draw_polygons(image, items, color="red", width=3, draw_label=True):
    draw = ImageDraw.Draw(image)
    for obj in items:
        poly = obj.get("polygon")
        if not poly: continue
        pts = [(float(x), float(y)) for x, y in poly]
        draw.polygon(pts, outline=color, width=width)
    return image

def surya_to_dict(obj):
    if hasattr(obj, "__dict__"):
        out = {}
        for k, v in obj.__dict__.items():
            if hasattr(v, "__dict__"):
                out[k] = surya_to_dict(v)
            elif isinstance(v, list):
                out[k] = [surya_to_dict(i) if hasattr(i,"__dict__") else i for i in v]
            else:
                out[k] = v
        return out
    return obj