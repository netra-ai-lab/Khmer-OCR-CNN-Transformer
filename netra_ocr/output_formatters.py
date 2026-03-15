# output_formatters.py
"""
Multi-format OCR output formatters.

Segment schema
--------------
Text segment:
    { "type":"text",  "text":str, "crop":PIL.Image,
      "bbox":(x1,y1,x2,y2), "label":str }

Image segment (Table / Picture / Formula):
    { "type":"image", "crop":PIL.Image,
      "bbox":(x1,y1,x2,y2), "label":str }

Format behaviour
----------------
Format  │ Text segments          │ Visual segments
────────┼────────────────────────┼─────────────────────────────
.txt    │ OCR string             │ [Label] placeholder
.md     │ OCR string + markup    │ [Label] blockquote
.html   │ <div> with OCR text    │ <img> of crop
.pdf    │ text via Khmer font;   │ image crop
        │ falls back to crop img │
        │ if no font available   │
.docx   │ floating text box,     │ floating anchored image
        │ Khmer font specified   │
"""

from __future__ import annotations

import io, os, base64
from html import escape as html_escape
from pathlib import Path
from typing import List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

# ── label helpers ──────────────────────────────────────────────────────────
VISUAL_LABELS   = {"Table", "Picture", "Figure", "Formula"}
BOLD_LABELS     = {"Title", "Section-header", "Page-header"}
ITALIC_LABELS   = {"Caption", "Footnote", "Page-footer"}
LIST_LABEL      = "List-item"

LABEL_FONT_SCALE = {
    "Title":          1.6,
    "Section-header": 1.25,
    "Page-header":    1.1,
    "Caption":        0.85,
    "Footnote":       0.75,
    "Page-footer":    0.75,
}

# Khmer font candidates (checked in order on the host system)
_KHMER_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/KhmerUI.ttf",
    "C:/Windows/Fonts/KhmerUIb.ttf",
    "C:/Windows/Fonts/leelawad.ttf",
    # Linux
    "/usr/share/fonts/truetype/khmeros/KhmerOS.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifKhmer-Regular.ttf",
    # Local project fonts/ directory
    os.path.join(os.path.dirname(__file__), "fonts", "KhmerOS.ttf"),
    os.path.join(os.path.dirname(__file__), "fonts", "NotoSansKhmer-Regular.ttf"),
]

# Preferred Khmer font name for DOCX (Word uses the system font by this name)
_DOCX_KHMER_FONTS = ["Khmer OS Siemreap", "Khmer OS", "Noto Sans Khmer", "Leelawadee UI"]


def _find_khmer_ttf() -> Optional[str]:
    for p in _KHMER_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _to_png_bytes(crop) -> bytes:
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _to_b64_uri(crop) -> str:
    return "data:image/png;base64," + base64.b64encode(_to_png_bytes(crop)).decode()


# ══════════════════════════════════════════════════════════════════════════════
#  TXT
# ══════════════════════════════════════════════════════════════════════════════
def save_txt(segments: List[dict], output_path: str) -> None:
    lines = []
    for seg in segments:
        if seg["type"] == "text":
            t = seg["text"].strip()
            if t:
                lines.append(t)
        else:
            lines.append(f"[{seg.get('label', 'Image')}]")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[Formatter] TXT  → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════
def save_markdown(segments: List[dict], output_path: str) -> None:
    out: List[str] = []
    for seg in segments:
        if seg["type"] == "image":
            out.append(f"> [{seg.get('label', 'Image')}]\n\n")
            continue
        text  = seg["text"].strip()
        if not text:
            continue
        label = seg.get("label", "Text")
        if label == "Title":
            out.append(f"# {text}\n\n")
        elif label == "Section-header":
            out.append(f"## {text}\n\n")
        elif label == "Page-header":
            out.append(f"### {text}\n\n")
        elif label == LIST_LABEL:
            out.append(f"- {text}\n")
        elif label in ITALIC_LABELS:
            out.append(f"*{text}*\n\n")
        else:
            out.append(f"{text}\n\n")
    Path(output_path).write_text("".join(out), encoding="utf-8")
    print(f"[Formatter] MD   → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  HTML
#  Text segments → <div> with actual OCR text, absolutely positioned
#  Visual segments → <img> of the crop, absolutely positioned
# ══════════════════════════════════════════════════════════════════════════════
def save_html(
    segments: List[dict],
    output_path: str,
    image_size: Tuple[int, int],
) -> None:
    img_w, img_h = image_size
    aspect = (img_h / img_w) * 100
    els: List[str] = []

    for seg in segments:
        bbox  = seg.get("bbox")
        label = seg.get("label", "Text")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        l = (x1 / img_w) * 100
        t = (y1 / img_h) * 100
        w = max(0.1, (x2 - x1) / img_w * 100)
        h = max(0.05, (y2 - y1) / img_h * 100)
        # font-size as % of page height so it scales with zoom
        line_h_pct = h * 0.72 * LABEL_FONT_SCALE.get(label, 1.0)

        if seg["type"] == "image":
            crop = seg.get("crop")
            if crop is None:
                continue
            els.append(
                f'<img class="vis {label.lower().replace("-","_")}"'
                f' src="{_to_b64_uri(crop)}" alt="{html_escape(label)}"'
                f' style="left:{l:.3f}%;top:{t:.3f}%;width:{w:.3f}%;height:{h:.3f}%;">'
            )
        else:
            text  = seg["text"].strip()
            if not text:
                continue
            bold   = "font-weight:700;" if label in BOLD_LABELS   else ""
            italic = "font-style:italic;" if label in ITALIC_LABELS else ""
            els.append(
                f'<div class="txt {label.lower().replace("-","_")}"'
                f' style="left:{l:.3f}%;top:{t:.3f}%;width:{w:.3f}%;height:{h:.3f}%;'
                f'font-size:{line_h_pct:.2f}%;{bold}{italic}">'
                f'{html_escape(text)}</div>'
            )

    html = f"""<!DOCTYPE html>
<html lang="km">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OCR Output</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{
      background:#888;
      display:flex; justify-content:center; padding:32px 16px;
      font-family:'Khmer OS','Khmer UI','Noto Sans Khmer','Leelawadee UI',sans-serif;
    }}
    .page {{
      position:relative; width:100%; max-width:{img_w}px;
      padding-bottom:{aspect:.3f}%;
      background:#fff; box-shadow:0 8px 40px rgba(0,0,0,.4);
    }}
    .txt, .vis {{
      position:absolute; overflow:hidden;
      line-height:1.2; white-space:pre-wrap; word-break:break-word;
      color:#111;
    }}
    .vis {{ object-fit:contain; }}
    .title        {{ color:#000; }}
    .section_header {{ border-bottom:1px solid #ccc; padding-bottom:1px; }}
    .caption, .footnote, .page_footer {{ color:#555; }}
  </style>
</head>
<body><div class="page">
{"".join(els)}
</div></body>
</html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"[Formatter] HTML → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PDF  —  Hybrid: background image + PIL text stamps + invisible text
#
#  Why this approach produces correct Khmer shaping
#  -------------------------------------------------
#  ReportLab writes Unicode codepoints directly into the PDF glyph stream
#  with no OpenType shaping engine, so coeng (U+17D2) stacking never happens.
#
#  Pillow's ImageDraw.text() calls FreeType directly.  On any system where
#  libraqm is available (Linux with raqm, Windows with a Khmer-capable font),
#  FreeType applies the GSUB + GPOS rules and produces correctly stacked
#  subscripts.  The resulting *rendered pixels* are then embedded as an image
#  in the PDF — shaping is baked into the pixels before ReportLab ever sees
#  the text, so ReportLab's lack of an OT shaping engine doesn't matter.
#
#  Pipeline (per text segment)
#  ---------------------------
#  1. Background layer   — draw the original scan so logos, seals, signatures
#                          and visual regions (tables, pictures) are preserved.
#  2. Erase layer        — white rectangle over each text bbox removes the
#                          original blurry/scanned text.
#  3. Text stamp layer   — 3× supersampled PIL image rendered with the Khmer
#                          TTF font; this image is drawn at the bbox position.
#  4. Invisible text     — zero-alpha ReportLab drawString makes the PDF
#                          searchable and copy-paste works correctly.
#
#  Visual segments (Table / Picture / Formula) are left untouched because
#  the background image already contains them at full quality.
# ══════════════════════════════════════════════════════════════════════════════

# Supersampling factor for PIL text stamps — higher = sharper but larger PDF
_STAMP_SCALE = 3


def _make_text_stamp(
    text: str,
    box_w: int,
    box_h: int,
    font_path: str,
    bold: bool = False,
    italic: bool = False,
) -> "PIL.Image.Image":
    """
    Render *text* into a transparent RGBA image of size (box_w × box_h).

    The image is rendered at _STAMP_SCALE × resolution for sharpness, then
    downsampled.  Font size is chosen so the text fits within the box.

    Pillow calls FreeType which (with libraqm) applies full OpenType shaping —
    Khmer coeng stacking, vowel marks, and diacritic positioning are all
    handled correctly at this stage.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    ss = _STAMP_SCALE
    canvas_w = max(1, box_w * ss)
    canvas_h = max(1, box_h * ss)

    # Start at 80% of box height and shrink until text fits
    font_size = max(8, int(canvas_h * 0.80))
    min_size  = 8
    font_obj  = None

    try:
        while font_size >= min_size:
            try:
                f = ImageFont.truetype(font_path, font_size)
            except Exception:
                f = ImageFont.load_default()
            probe = ImageDraw.Draw(PILImage.new("RGBA", (1, 1)))
            tb = probe.textbbox((0, 0), text, font=f)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            if tw <= canvas_w * 0.98 and th <= canvas_h * 0.98:
                font_obj = f
                break
            font_size -= 1
        if font_obj is None:
            font_obj = ImageFont.load_default()
    except Exception:
        font_obj = ImageFont.load_default()

    # Draw onto transparent canvas, left-aligned, vertically centred
    stamp = PILImage.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    draw  = ImageDraw.Draw(stamp)

    tb = draw.textbbox((0, 0), text, font=font_obj)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    x = 0                               # left-aligned (matches document layout)
    y = max(0, (canvas_h - th) // 2) - tb[1]   # vertically centred
    draw.text((x, y), text, font=font_obj, fill=(0, 0, 0, 255))

    # Downsample to target size with LANCZOS for clean sub-pixel rendering
    stamp = stamp.resize((box_w, box_h), PILImage.LANCZOS)
    return stamp


def save_pdf(
    segments: List[dict],
    output_path: str,
    image_size: Tuple[int, int],
    image_path: Optional[str] = None,
) -> None:
    """
    Generate a PDF with correctly shaped Khmer text.

    Layers (bottom → top)
    ─────────────────────
    1. Original scan image  — preserves logos, seals, stamps, visual regions
    2. White erase rect     — hides blurry original text at each text bbox
    3. PIL text stamp       — OCR text rendered via FreeType (correct shaping)
    4. Invisible text       — zero-alpha text makes the file searchable

    Args:
        segments:   Segment list from the OCR pipeline.
        output_path: Destination .pdf path.
        image_size:  (width, height) of the source image in pixels.
        image_path:  Path to the original source image.  Required for the
                     background layer.  If None or missing, the background
                     is omitted and text bboxes are drawn on a white page.
    """
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.colors import Color, white
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    img_w, img_h = image_size

    # PDF page dimensions = source image dimensions (1 pt = 1 px).
    # This eliminates all coordinate scaling — bbox pixels map directly to
    # PDF points with a single Y-axis flip (PDF Y=0 is bottom-left).
    page_w, page_h = float(img_w), float(img_h)

    # ── font setup ─────────────────────────────────────────────────────────
    font_path = _find_khmer_ttf()
    if font_path:
        try:
            pdfmetrics.registerFont(TTFont("KhmerInvis", font_path))
            invis_font = "KhmerInvis"
            print(f"  [PDF] Invisible text font: {font_path}")
        except Exception:
            invis_font = "Helvetica"
    else:
        invis_font = "Helvetica"
        print("  [PDF] No Khmer font found — invisible text layer uses Helvetica")

    c = rl_canvas.Canvas(output_path, pagesize=(page_w, page_h))

    # ── Layer 1: background image ───────────────────────────────────────────
    has_bg = image_path and os.path.exists(str(image_path))
    if has_bg:
        c.drawImage(str(image_path), 0, 0, width=page_w, height=page_h)
    else:
        # No background: fill white so visual segments show their crop
        c.setFillColor(white)
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
        # Draw visual (image) segments explicitly when no background is available
        for seg in segments:
            if seg["type"] != "image":
                continue
            bbox = seg.get("bbox")
            crop = seg.get("crop")
            if not bbox or crop is None:
                continue
            x1, y1, x2, y2 = bbox
            pdf_y = page_h - y2          # flip Y axis
            try:
                c.drawImage(
                    ImageReader(io.BytesIO(_to_png_bytes(crop))),
                    float(x1), float(pdf_y),
                    width=float(x2 - x1), height=float(y2 - y1),
                    preserveAspectRatio=False, mask="auto",
                )
            except Exception as exc:
                print(f"  [PDF] skipped visual segment at {bbox}: {exc}")

    # ── Layers 2–4: process each text segment ──────────────────────────────
    ERASE_PAD = 2   # extra pixels around bbox for the white erase rect

    for seg in segments:
        if seg["type"] != "text":
            continue

        text  = seg["text"].strip()
        bbox  = seg.get("bbox")
        label = seg.get("label", "Text")
        if not text or not bbox:
            continue

        x1, y1, x2, y2 = bbox
        box_w = max(1, int(x2 - x1))
        box_h = max(1, int(y2 - y1))
        # PDF Y-axis is bottom-up; y1/y2 are measured from the top of the image
        pdf_y_bottom = page_h - y2      # bottom-left anchor for ReportLab

        # ── Layer 2: white erase ───────────────────────────────────────────
        c.setFillColor(white)
        c.setStrokeColor(white)
        c.rect(
            x1 - ERASE_PAD,
            pdf_y_bottom - ERASE_PAD,
            box_w + ERASE_PAD * 2,
            box_h + ERASE_PAD * 2,
            fill=1, stroke=1,
        )

        # ── Layer 3: PIL text stamp (correctly shaped Khmer pixels) ────────
        if font_path:
            try:
                bold   = label in BOLD_LABELS
                italic = label in ITALIC_LABELS
                stamp  = _make_text_stamp(
                    text, box_w, box_h, font_path,
                    bold=bold, italic=italic,
                )
                buf = io.BytesIO()
                stamp.save(buf, format="PNG")
                buf.seek(0)
                c.drawImage(
                    ImageReader(buf),
                    float(x1), float(pdf_y_bottom),
                    width=float(box_w), height=float(box_h),
                    mask="auto",
                )
            except Exception as exc:
                print(f"  [PDF] text stamp failed at {bbox}: {exc}")
        else:
            # No font available — keep the crop image so something is visible
            crop = seg.get("crop")
            if crop:
                try:
                    c.drawImage(
                        ImageReader(io.BytesIO(_to_png_bytes(crop))),
                        float(x1), float(pdf_y_bottom),
                        width=float(box_w), height=float(box_h),
                        preserveAspectRatio=False, mask="auto",
                    )
                except Exception:
                    pass

        # ── Layer 4: invisible text (searchable / copy-paste) ──────────────
        # alpha=0 makes it transparent; the font must be registered or
        # ReportLab will use Helvetica which still encodes the codepoints.
        invis_sz = max(4.0, box_h * 0.70)
        try:
            c.setFont(invis_font, invis_sz)
        except Exception:
            c.setFont("Helvetica", invis_sz)
        c.setFillColor(Color(0, 0, 0, alpha=0))
        try:
            c.drawString(float(x1), float(pdf_y_bottom) + box_h * 0.15, text)
        except Exception:
            pass

    c.save()
    print(f"[Formatter] PDF  → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DOCX
#  Text segments → floating text boxes with explicit Khmer font name
#                  (Word resolves the font from the system)
#  Visual segments → floating anchored images
# ══════════════════════════════════════════════════════════════════════════════
def save_docx(
    segments: List[dict],
    output_path: str,
    image_size: Tuple[int, int],
) -> None:
    try:
        from docx import Document
        from docx.shared import Emu
        from docx.opc.part import Part
        from docx.opc.packuri import PackURI
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        import lxml.etree as etree
    except ImportError:
        raise ImportError("pip install python-docx")

    img_w, img_h = image_size
    PAGE_W, PAGE_H = 11_906_400, 16_838_400   # A4 in EMU
    MARGIN    = 720_720
    CONTENT_W = PAGE_W - 2 * MARGIN
    CONTENT_H = PAGE_H - 2 * MARGIN
    scale     = min(CONTENT_W / img_w, CONTENT_H / img_h)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width    = Emu(PAGE_W);  sec.page_height   = Emu(PAGE_H)
    sec.left_margin   = sec.right_margin  = Emu(MARGIN)
    sec.top_margin    = sec.bottom_margin = Emu(MARGIN)
    for p in doc.paragraphs:
        p._element.getparent().remove(p._element)

    draw_id   = 1
    img_index = 1

    # Choose the best available Khmer font name for Word to resolve
    khmer_font = _DOCX_KHMER_FONTS[0]   # "Khmer UI" — ships with Windows 8+

    def _anchor_head(emu_x, emu_y, emu_w, emu_h, did):
        return (
            f'<wp:anchor simplePos="0" relativeHeight="251658240" behindDoc="0"'
            f' locked="0" layoutInCell="1" allowOverlap="1"'
            f' distT="0" distB="0" distL="0" distR="0">'
            f'<wp:simplePos x="0" y="0"/>'
            f'<wp:positionH relativeFrom="page"><wp:posOffset>{emu_x}</wp:posOffset></wp:positionH>'
            f'<wp:positionV relativeFrom="page"><wp:posOffset>{emu_y}</wp:posOffset></wp:positionV>'
            f'<wp:extent cx="{emu_w}" cy="{emu_h}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:wrapNone/>'
            f'<wp:docPr id="{did}" name="E{did}"/>'
        )

    for seg in segments:
        bbox = seg.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        emu_x = int(MARGIN + x1 * scale)
        emu_y = int(MARGIN + y1 * scale)
        emu_h = max(9_144, int((y2 - y1) * scale))
        label = seg.get("label", "Text")

        if seg["type"] == "image":
            # Images: preserve exact detected bbox width
            emu_w = max(9_144, int((x2 - x1) * scale))
        else:
            # Text: extend to right content boundary.
            # Surya bboxes are tight ink-pixel bounds. When Word renders Khmer UI
            # at the computed font size, even slightly wider glyph metrics cause
            # text to wrap inside the narrow box. spAutoFit then grows the box
            # downward, pushing it into the next element → clutter and overlap.
            # Extending to the right margin gives the text all available horizontal
            # space, completely preventing wrapping for single-column documents.
            emu_w = max(9_144, PAGE_W - MARGIN - emu_x)

        # ── visual segment → anchored image ───────────────────────────────
        if seg["type"] == "image":
            crop = seg.get("crop")
            if crop is None:
                continue
            png   = _to_png_bytes(crop)
            puri  = PackURI(f"/word/media/img_{img_index:04d}.png")
            part  = Part(puri, "image/png", png, doc.part.package)
            rId   = doc.part.relate_to(part, RT.IMAGE)
            img_index += 1

            xml = (
                f'<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
                f' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
                f' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
                f' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
                f' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<w:r><w:drawing>'
                + _anchor_head(emu_x, emu_y, emu_w, emu_h, draw_id) +
                f'<wp:cNvGraphicFramePr>'
                f'<a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
                f'<a:graphic><a:graphicData'
                f' uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                f'<pic:pic>'
                f'<pic:nvPicPr><pic:cNvPr id="{draw_id}" name="E{draw_id}"/>'
                f'<pic:cNvPicPr/></pic:nvPicPr>'
                f'<pic:blipFill>'
                f'<a:blip r:embed="{rId}"/>'
                f'<a:stretch><a:fillRect/></a:stretch>'
                f'</pic:blipFill>'
                f'<pic:spPr>'
                f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{emu_w}" cy="{emu_h}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
                f'</pic:spPr>'
                f'</pic:pic></a:graphicData></a:graphic>'
                f'</wp:anchor></w:drawing></w:r></w:p>'
            )

        # ── text segment → floating text box ──────────────────────────────
        else:
            text = seg["text"].strip()
            if not text:
                continue

            bold     = label in BOLD_LABELS
            italic   = label in ITALIC_LABELS
            # Derive font size from bbox pixel height relative to image height,
            # then project onto A4 page points.
            #
            # OLD (buggy): half_pt = f(emu_h)
            #   emu_h = bbox_px × scale, and scale can be 17 000+ EMU/px for small
            #   images, so emu_h becomes huge → font formula produces 28–36pt body
            #   text → characters are wider than the tight bbox → text wraps.
            #
            # NEW: font_sz ∝ (bbox_px / img_h) × A4_page_height_pt
            #   This is scale-independent: a line that is 3% of the image height
            #   maps to 3% of the A4 page height regardless of image resolution.
            _A4_PT = 841.89
            line_h_frac = (y2 - y1) / img_h
            font_sz_pt  = max(7.0, min(20.0,
                line_h_frac * _A4_PT * 0.60
                * LABEL_FONT_SCALE.get(label, 1.0)))
            half_pt = int(font_sz_pt * 2)
            safe_t   = xml_escape(text)

            rpr_bold   = "<w:b/><w:bCs/>"         if bold   else ""
            rpr_italic = "<w:i/><w:iCs/>"          if italic else ""
            rpr = (
                f'<w:rFonts w:ascii="{khmer_font}" w:hAnsi="{khmer_font}"'
                f' w:cs="{khmer_font}"/>'
                f'<w:lang w:bidi="km"/>'
                f'{rpr_bold}{rpr_italic}'
                f'<w:sz w:val="{half_pt}"/><w:szCs w:val="{half_pt}"/>'
            )

            xml = (
                f'<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
                f' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
                f' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
                f' xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
                f' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
                f' xmlns:v="urn:schemas-microsoft-com:vml">'
                f'<w:r><w:rPr><w:noProof/></w:rPr>'
                f'<mc:AlternateContent><mc:Choice Requires="wps">'
                f'<w:drawing>'
                + _anchor_head(emu_x, emu_y, emu_w, emu_h, draw_id) +
                f'<wp:cNvGraphicFramePr/>'
                f'<a:graphic><a:graphicData'
                f' uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
                f'<wps:wsp>'
                f'<wps:cNvSpPr txBox="1"><a:spLocks noChangeArrowheads="1"/></wps:cNvSpPr>'
                f'<wps:spPr>'
                f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{emu_w}" cy="{emu_h}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
                f'<a:noFill/><a:ln><a:noFill/></a:ln>'
                f'</wps:spPr>'
                f'<wps:txbx><w:txbxContent>'
                f'<w:p><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr>'
                f'<w:r><w:rPr>{rpr}</w:rPr>'
                f'<w:t xml:space="preserve">{safe_t}</w:t>'
                f'</w:r></w:p>'
                f'</w:txbxContent></wps:txbx>'
                # spAutoFit: Word expands the textbox height to fit rendered
                # content, so tall Khmer stacked diacritics / subscripts /
                # superscripts are never clipped.
                # insT/insB=0 removes vertical padding so the baseline sits
                # where the detected bbox expects it.
                f'<wps:bodyPr anchor="t" insL="45720" insR="45720"'
                f' insT="0" insB="0">'
                f'<a:spAutoFit/>'
                f'</wps:bodyPr>'
                f'</wps:wsp></a:graphicData></a:graphic>'
                f'</wp:anchor></w:drawing>'
                f'</mc:Choice>'
                f'<mc:Fallback><w:pict><v:textbox>'
                f'<w:txbxContent><w:p><w:r>'
                f'<w:t xml:space="preserve">{safe_t}</w:t>'
                f'</w:r></w:p></w:txbxContent>'
                f'</v:textbox></w:pict></mc:Fallback>'
                f'</mc:AlternateContent></w:r></w:p>'
            )

        try:
            doc.element.body.append(etree.fromstring(xml))
            draw_id += 1
        except Exception as exc:
            print(f"  [DOCX] skipped segment at {bbox}: {exc}")

    doc.save(output_path)
    print(f"[Formatter] DOCX → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Dispatcher
# ══════════════════════════════════════════════════════════════════════════════
_FORMAT_MAP = {
    ".txt":  save_txt,
    ".md":   save_markdown,
    ".html": save_html,
    ".htm":  save_html,
    ".pdf":  save_pdf,
    ".docx": save_docx,
}
SUPPORTED_FORMATS = list(_FORMAT_MAP.keys())


def save_output(
    segments: List[dict],
    output_path: str,
    image_size: Optional[Tuple[int, int]] = None,
    image_path: Optional[str] = None,
) -> None:
    ext = Path(output_path).suffix.lower()
    fn  = _FORMAT_MAP.get(ext)
    if fn is None:
        print(f"[Formatter] Unknown extension '{ext}' — falling back to .txt")
        fn, output_path = save_txt, str(Path(output_path).with_suffix(".txt"))

    import inspect
    sig_params = inspect.signature(fn).parameters
    kwargs: dict = {}
    if "image_size" in sig_params:
        if image_size is None:
            raise ValueError(f"image_size=(w,h) required for '{ext}'")
        kwargs["image_size"] = image_size
    if "image_path" in sig_params:
        kwargs["image_path"] = image_path

    fn(segments, output_path, **kwargs)