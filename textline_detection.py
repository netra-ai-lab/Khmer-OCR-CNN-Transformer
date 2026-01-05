from PIL import Image, ImageDraw, ImageFont
from surya.detection import DetectionPredictor

# ---------------------------------------------------------
# TEXTLINE EXTRACTION HELPERS (MODIFIED)
# ---------------------------------------------------------
def extract_textline_crops(image, textline_pred, expansion_px=5, padding_px=10):
    """
    Given Surya textline prediction:
    1. EXPAND the bounding box (to catch character edges).
    2. CROP the image.
    3. PAD with white background (to give OCR breathing room).
    """
    crops = []
    img_w, img_h = image.size

    for obj in textline_pred.bboxes:
        poly = obj.polygon
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]

        # 1. Get Base Coordinates
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))

        # 2. EXPAND BOUNDING BOX (Grab context from original image)
        x0 = max(0, x0 - expansion_px)
        y0 = max(0, y0 - expansion_px)
        x1 = min(img_w, x1 + expansion_px)
        y1 = min(img_h, y1 + expansion_px)

        # Skip invalid crops
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue

        # 3. Crop
        crop = image.crop((x0, y0, x1, y1))

        # 4. ADD WHITE PADDING
        if padding_px > 0:
            new_w = crop.width + (padding_px * 2)
            new_h = crop.height + (padding_px * 2)
            
            # Create white background (RGB)
            padded_crop = Image.new("RGB", (new_w, new_h), (255, 255, 255))
            # Paste crop in center
            padded_crop.paste(crop, (padding_px, padding_px))
            crop = padded_crop

        # Return crop and ORIGINAL bbox (for sorting)
        crops.append((crop, (x0, y0, x1, y1)))

    return crops


def run_textline_detector(image_path, expansion_px=5, padding_px=10):
    """
    Lightweight function:
    - load surya predictors
    - detect textlines
    - return sorted textline boxes + PADDED crops
    """
    image = Image.open(image_path).convert("RGB")

    det_predictor = DetectionPredictor()
    textline_pred = det_predictor([image])[0]

    # Extract crops with padding
    crops = extract_textline_crops(
        image, 
        textline_pred, 
        expansion_px=expansion_px, 
        padding_px=padding_px
    )

    # Sort top-to-bottom based on Y-coordinate
    crops_sorted = sorted(crops, key=lambda x: x[1][1])

    return crops_sorted, textline_pred

# ---------------------------------------------------------
# VISUALIZATION UTILS (Optional, kept for compatibility)
# ---------------------------------------------------------
try:
    FONT = ImageFont.truetype("arial.ttf", 16)
except:
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
                out[k] = [surya_to_dict(i) if hasattr(i, "__dict__") else i for i in v]
            else:
                out[k] = v
        return out
    return obj