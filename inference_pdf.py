import os
import io
from PIL import Image, ImageOps, ImageDraw, ImageFont

# Surya Imports (Layout & Detection)
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.foundation import FoundationPredictor
from surya.settings import settings

# PDF Generation Imports (ReportLab)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import Color, white
from reportlab.lib.utils import ImageReader

# Import Inference Class
from inference import KhmerOCRInference
from crnn_se_model import KhmerOCR

# ==============================================================================
# 1. HELPER FUNCTIONS (Geometry & Cropping)
# ==============================================================================

def get_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def is_center_inside(inner_bbox, outer_bbox):
    cx, cy = get_center(inner_bbox)
    ox0, oy0, ox1, oy1 = outer_bbox
    return ox0 <= cx <= ox1 and oy0 <= cy <= oy1

def crop_with_padding(image, bbox, expansion=5, padding=10):
    img_w, img_h = image.size
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - expansion)
    y0 = max(0, y0 - expansion)
    x1 = min(img_w, x1 + expansion)
    y1 = min(img_h, y1 + expansion)
    crop = image.crop((x0, y0, x1, y1))
    crop = ImageOps.expand(crop, border=padding, fill='white')
    return crop

# ==============================================================================
# 2. LAYOUT-AWARE PIPELINE
# ==============================================================================

def run_layout_aware_pipeline(image_path, ocr_model):
    print(f"🚀 Processing Image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # Initialize Surya predictors
    det_predictor = DetectionPredictor()
    layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))

    print("   Running Surya Detection & Layout...")
    textline_pred = det_predictor([image])[0]
    layout_pred = layout_predictor([image])[0]

    text_lines = [{"bbox": l.bbox} for l in textline_pred.bboxes]
    layout_blocks = sorted(layout_pred.bboxes, key=lambda b: (b.bbox[1], b.bbox[0]))

    structured_data = []
    assigned_line_indices = set()

    # Map text lines to specific layout blocks (Paragraphs, Headers, etc.)
    for block in layout_blocks:
        lines_in_block = []
        for i, line in enumerate(text_lines):
            if i in assigned_line_indices: continue
            if is_center_inside(line['bbox'], block.bbox):
                lines_in_block.append(line)
                assigned_line_indices.add(i)
        
        lines_in_block.sort(key=lambda l: l['bbox'][1])
        if lines_in_block:
            structured_data.append({"type": block.label, "bbox": block.bbox, "lines": lines_in_block})

    # Catch any lines not inside a layout block (Orphans)
    orphan_lines = [l for i, l in enumerate(text_lines) if i not in assigned_line_indices]
    if orphan_lines:
        orphan_lines.sort(key=lambda l: l['bbox'][1])
        structured_data.append({"type": "Orphan", "bbox": None, "lines": orphan_lines})

    print("   Running Recognition...")
    final_output = []
    
    # Run OCR on the crops
    for block in structured_data:
        for line in block['lines']:
            # Crop the line from the original image
            crop = crop_with_padding(image, line['bbox'], expansion=5, padding=10)
            
            # Use the imported OCR model
            text = ocr_model.predict(crop, beam_width=3)
            
            print(f"      {text}")
            final_output.append({"type": block['type'], "text": text, "bbox": line['bbox']})

    return final_output

# ==============================================================================
# 3. PDF GENERATION (Hybrid: Image Visuals + Invisible Text)
# ==============================================================================

def create_high_res_text_stamp(text, target_w, target_h, font_path):
    """Generates a high-res PIL image of text (Correct Visuals)."""
    scale = 3 # 3x High Res for sharpness
    canvas_w = int(target_w * scale)
    canvas_h = int(target_h * scale)
    
    img = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Heuristic font sizing
    font_size = int(canvas_h * 0.8) 
    min_size = 10
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Fit text to box
    while font_size > min_size:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        # Check if text fits with margin
        if (bbox[2]-bbox[0] < canvas_w * 0.95) and (bbox[3]-bbox[1] < canvas_h * 0.95):
            break
        font_size -= 2

    # Draw Text centered
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_w - text_w) // 2
    y = (canvas_h - text_h) // 2 - bbox[1]

    draw.text((x, y), text, font=font, fill="black")
    return img

def generate_clean_pdf(image_path, layout_results, output_path, font_path):
    print(f"🔨 Generating Final PDF: {output_path}")
    
    try:
        pdfmetrics.registerFont(TTFont('KhmerFont', font_path))
    except:
        print("❌ Error: Font not found. Copy/Paste might fail.")

    img = Image.open(image_path)
    width, height = img.size
    c = canvas.Canvas(output_path, pagesize=(width, height))

    # 1. Background Image (Keeps Logos, Seals, Signatures)
    c.drawImage(image_path, 0, 0, width=width, height=height)

    for item in layout_results:
        text = item['text']
        x0, y0, x1, y1 = item['bbox']
        
        box_w = x1 - x0
        box_h = y1 - y0
        pdf_y = height - y1

        # --- 2. THE "ERASER" (White Box) ---
        # Covers the original blurry text
        erase_padding = 2 
        c.setFillColor(white)
        c.setStrokeColor(white)
        c.rect(
            x0 - erase_padding, 
            pdf_y - erase_padding, 
            box_w + (erase_padding * 2), 
            box_h + (erase_padding * 2), 
            fill=1, 
            stroke=1
        )

        # --- 3. VISUAL TEXT (High-Res Image Stamp) ---
        # This renders the text cleanly as an image overlay
        text_stamp = create_high_res_text_stamp(text, box_w, box_h, font_path)
        img_buffer = io.BytesIO()
        text_stamp.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        c.drawImage(ImageReader(img_buffer), x0, pdf_y, width=box_w, height=box_h, mask='auto')

        # --- 4. INVISIBLE TEXT (For Copy-Paste) ---
        # This makes the PDF searchable and selectable
        c.setFillColor(Color(0, 0, 0, alpha=0)) 
        c.setFont('KhmerFont', box_h * 0.7)
        c.drawString(x0, pdf_y + (box_h * 0.15), text)

    c.save()
    print(f"✅ Success! PDF Saved.")

# ==============================================================================
# 4. MAIN
# ==============================================================================

if __name__ == "__main__":
    # --- CONFIG ---
    IMAGE_PATH = "khmer_document_4.jpg"
    MODEL_PATH = "./checkpoints/khmerocr_vgg_lstm_epoch100.pth"
    CHAR2IDX_PATH = "char2idx.json"
    FONT_PATH = "./fonts/KantumruyPro-Regular.ttf"
    
    RESULT_FOLDER = "results"
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    try:
        # 1. Load OCR Model (Using Class from inference.py)
        ocr_model = KhmerOCRInference(MODEL_PATH, CHAR2IDX_PATH, model_class=KhmerOCR, emb_dim=384)
        
        # 2. Process Image (Layout Analysis + OCR)
        results = run_layout_aware_pipeline(IMAGE_PATH, ocr_model)
        
        # 3. Generate PDF
        base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        pdf_path = os.path.join(RESULT_FOLDER, f"{base_name}_final.pdf")
        
        if os.path.exists(FONT_PATH):
            generate_clean_pdf(IMAGE_PATH, results, pdf_path, FONT_PATH)
        else:
            print("⚠️ Font path not found. Skipping PDF generation.")

        print("\n✅ Processing complete.")

    except Exception as e:
        import traceback
        traceback.print_exc()