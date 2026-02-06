import cv2
import numpy as np
from PIL import Image
from .config import Config

def get_iou(box1, box2):
    """Calculates Intersection over Union (IoU) to find overlaps."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Intersection coordinates
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    
    # We use "Intersection over Smallest Box" (IoS) 
    # This is better for finding if one box is "inside" another
    return intersection / min(area1, area2)

def suppress_overlapping_boxes(boxes, overlap_threshold=0.7):
    """Merges boxes that are stacked or highly overlapping."""
    if not boxes: return []
    
    # Sort boxes by area (largest first)
    boxes = sorted(boxes, key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
    
    keep = []
    merged_indices = set()

    for i in range(len(boxes)):
        if i in merged_indices: continue
        
        current_box, current_cls = boxes[i]
        
        for j in range(i + 1, len(boxes)):
            if j in merged_indices: continue
            
            compare_box, compare_cls = boxes[j]
            
            # Check overlap
            if get_iou(current_box, compare_box) > overlap_threshold:
                # Merge the boxes: Take the outer boundaries
                current_box = [
                    min(current_box[0], compare_box[0]),
                    min(current_box[1], compare_box[1]),
                    max(current_box[2], compare_box[2]),
                    max(current_box[3], compare_box[3])
                ]
                merged_indices.add(j)
        
        keep.append((current_box, current_cls))
        
    return keep

def is_graphical_line(binary_crop):
    """
    Detects if the detected ink is a solid graphical line.
    Relaxed aspect ratio to prevent thin text from being ignored.
    """
    h, w = binary_crop.shape[:2]
    if h == 0 or w == 0: return False

    ink_pixels = cv2.countNonZero(binary_crop)
    density = ink_pixels / (w * h)

    # Text is rarely 50x wider than it is tall, but graphical lines are.
    is_extremely_thin = (w / h > 50) or (h / w > 50)

    # Graphical lines are usually very solid (ink density > 80%)
    is_very_dense = density > 0.80

    return is_extremely_thin and is_very_dense

def validate_non_text_content(img_np, box):
    """
    Checks non-text boxes (Pictures, Tables) to see if they are actually empty.
    Returns: The box if valid, None if empty.
    """
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)

    if (x2 - x1) < 5 or (y2 - y1) < 5: return None

    crop = img_np[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop

    # 1. BRIGHTNESS CHECK
    # If it's purely white/background (mean > 250), it's empty.
    if np.mean(gray) > 252:
        return None

    # 2. VARIANCE CHECK
    if np.std(gray) < 5:
        return None

    # 3. EDGE CHECK (Canny)
    # A picture or table must have edges.
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = cv2.countNonZero(edges)
    
    # If less than 1% of the area is edges, it's likely noise or a blank box
    if edge_pixels < ((x2-x1)*(y2-y1) * 0.005):
        return None

    return [x1, y1, x2, y2]

def analyze_content_type(binary_crop):
    """
    Advanced structural analysis specifically for Khmer script.
    Detects if a 'Picture/Table' box is actually a Title, Header, or Small Word.
    """
    h, w = binary_crop.shape[:2]
    if h < 5 or w < 5: return 'text'
    
    # 1. Connected Component Analysis (The Blob Test)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_crop)
   
    valid_blobs = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] > 4]
    num_blobs = len(valid_blobs)

    # 2. Aspect Ratio (Khmer words/lines are long)
    aspect_ratio = w / h
    
    # 3. Horizontal Projection Profile
    row_sums = np.sum(binary_crop, axis=1)
    ink_rows = row_sums > (np.max(row_sums) * 0.1)
    transitions = np.sum(np.diff(ink_rows.astype(int)) != 0)

    # Case A: It's clearly a paragraph (multiple lines)
    if transitions >= 3: 
        return 'text'

    # Case B: It's a single word or short line (like "ស្ដីពី")
    if transitions <= 2:
        # Text is almost always wider than 1.5x height
        if aspect_ratio > 1.3:
            # If there are at least 2 blobs (consonant + vowel/diacritic), it's text
            if num_blobs >= 2:
                return 'text'
            # Extremely wide single blobs are usually underlined text or lines
            if aspect_ratio > 4.0:
                return 'text'

    # Case C: Density Check (Pictures/Logos are heavy, Text is airy)
    density = cv2.countNonZero(binary_crop) / (w * h)
    if density > 0.70 and aspect_ratio < 2.0:
        return 'picture'

    # Case D: Very small boxes that aren't square
    if h < 30 and aspect_ratio > 1.5:
        return 'text'

    return 'picture'

def snap_to_ink(img_np, box, padding=3, min_ink_pixels=5, lookahead=15, expand_y=5):
    """
    Dynamically expands the bounding box left and right until no more ink is found.
    
    Args:
        lookahead: How many empty pixels to skip before deciding the line has ended.
                   (Crucial for Khmer to bridge gaps between characters).
        expand_y: Vertical search buffer to ensure we catch tall/low strokes.
    """
    x1, y1, x2, y2 = map(int, box)
    img_h, img_w = img_np.shape[:2]

    # 1. Prepare a slightly taller vertical strip for searching
    search_y1 = max(0, y1 - expand_y)
    search_y2 = min(img_h, y2 + expand_y)
    
    # 2. Convert a horizontal band to binary for fast pixel checking
    # We crop the full width of the image at the specific height of this line
    full_band = img_np[search_y1:search_y2, 0:img_w]
    gray = cv2.cvtColor(full_band, cv2.COLOR_RGB2GRAY) if len(full_band.shape) == 3 else full_band
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean noise (ignore tiny speckles)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Helper: Check if a specific column in our band has ink
    def col_has_ink(x_coord):
        if x_coord < 0 or x_coord >= img_w: return False
        return cv2.countNonZero(binary[:, x_coord]) > 0

    # 3. EXPAND LEFT
    curr_x1 = x1
    empty_count = 0
    while curr_x1 > 0:
        if col_has_ink(curr_x1 - 1):
            curr_x1 -= 1
            empty_count = 0 # Reset if ink found
        else:
            empty_count += 1
            curr_x1 -= 1
        
        if empty_count >= lookahead:
            curr_x1 += empty_count # Backtrack to where ink actually ended
            break

    # 4. EXPAND RIGHT
    curr_x2 = x2
    empty_count = 0
    while curr_x2 < img_w:
        if col_has_ink(curr_x2):
            curr_x2 += 1
            empty_count = 0
        else:
            empty_count += 1
            curr_x2 += 1
            
        if empty_count >= lookahead:
            curr_x2 -= empty_count # Backtrack
            break

    # 5. VERTICAL REFINEMENT
    # Now that we have the full horizontal width, shrink top/bottom strictly to ink
    final_crop = binary[:, curr_x1:curr_x2]
    if final_crop.size == 0 or cv2.countNonZero(final_crop) < min_ink_pixels:
        return None, False
    
    coords = cv2.findNonZero(final_crop)
    bx, by, bw, bh = cv2.boundingRect(coords)

    # 6. CALCULATE FINAL COORDINATES with padding
    # Padding on X is full, Padding on Y is halved to keep it tight
    res_x1 = max(0, curr_x1 + bx - padding)
    res_y1 = max(0, search_y1 + by - (padding // 2))
    res_x2 = min(img_w, curr_x1 + bx + bw + padding)
    res_y2 = min(img_h, search_y1 + by + bh + (padding // 2))

    # Check for graphical line (e.g. underline or border)
    if is_graphical_line(final_crop):
        return None, True

    return [res_x1, res_y1, res_x2, res_y2], False

def extract_layout_elements(img, segmentation_map, pred_heatmap):
    """
    Args:
        img: PIL Image
        segmentation_map: argmax result (classes)
        pred_heatmap: max probability map (0.0 to 1.0)
    """
    
    img_np = np.array(img.convert("RGB"))

    all_content_mask = (segmentation_map > Config.ENTRY_THRESHOLD).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, Config.MORPH_KERNEL_SIZE)
    morphed_mask = cv2.morphologyEx(all_content_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 1 or w < 2: continue

        # Confidence Scoring logic
        mask_cnt = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_cnt, [cnt - (x, y)], -1, 255, -1)

        score = cv2.mean(pred_heatmap[y:y+h, x:x+w], mask=mask_cnt)[0]

        if score < Config.SCORE_THRESHOLD:
            continue

        components.append([x, y, x + w, y + h])

    if not components: return [], []

    # Vertical grouping
    components.sort(key=lambda b: b[1])
    line_clusters = []
    while components:
        curr = components.pop(0)
        bx1, by1, bx2, by2 = curr
        matched = False
        for cluster in line_clusters:
            lx1, ly1, lx2, ly2 = cluster[-1]
            inter_y = max(0, min(by2, ly2) - max(by1, ly1))
            min_h = min(by2-by1, ly2-ly1)
            if min_h > 0 and (inter_y / min_h) > Config.LINE_OVERLAP_THRESHOLD:
                cluster.append(curr)
                matched = True
                break
        if not matched: line_clusters.append([curr])

    intermediate_results = []

    # Arbitration (Confidence-Weighted Logic)
    for cluster in line_clusters:
        line_x1, line_y1 = min(b[0] for b in cluster), min(b[1] for b in cluster)
        line_x2, line_y2 = max(b[2] for b in cluster), max(b[3] for b in cluster)

        line_seg_map = segmentation_map[line_y1:line_y2, line_x1:line_x2]
        line_conf_map = pred_heatmap[line_y1:line_y2, line_x1:line_x2]

        content_mask = line_seg_map > 0
        if not np.any(content_mask): continue

        pixels_in_line = line_seg_map[content_mask]
        conf_values = line_conf_map[content_mask]

        class_weighted_mass = {}
        for cls_id, conf in zip(pixels_in_line, conf_values):
            class_weighted_mass[cls_id] = class_weighted_mass.get(cls_id, 0.0) + conf

        total_mass = sum(class_weighted_mass.values())

        # Filter: Only consider classes that own significant portion of mass
        significant_classes = [
            cls for cls, mass in class_weighted_mass.items()
            if (mass / total_mass) > Config.SIGNIFICANCE_THRESHOLD
        ]

        if significant_classes:
            dominant_class = max(significant_classes, key=lambda c: class_weighted_mass[c])
        else:
            dominant_class = max(class_weighted_mass, key=class_weighted_mass.get)

        # Horizontal Merging
        cluster.sort(key=lambda b: b[0])
        curr_x1, curr_y1, curr_x2, curr_y2 = cluster[0]
        for i in range(1, len(cluster)):
            nx1, ny1, nx2, ny2 = cluster[i]
            if nx1 - curr_x2 < Config.MERGE_X_DIST:
                curr_x1, curr_y1 = min(curr_x1, nx1), min(curr_y1, ny1)
                curr_x2, curr_y2 = max(curr_x2, nx2), max(curr_y2, ny2)
            else:
                intermediate_results.append(((curr_x1, curr_y1, curr_x2, curr_y2), dominant_class))
                curr_x1, curr_y1, curr_x2, curr_y2 = nx1, ny1, nx2, ny2
        intermediate_results.append(((curr_x1, curr_y1, curr_x2, curr_y2), dominant_class))

    # 5. Final Output Generation
    output_boxes = []
    output_crops = []

    raw_results = []
    
    text_classes = [1, 2, 4, 5, 6, 8, 10, 11] 
    
    # IDs that the AI might hallucinate for text (Formula, Picture, Table)
    picture_classes = [3, 7, 9]

    for (box, cls_id) in intermediate_results:
        refined_box = None
        target_cls = cls_id
        
        # Binary crop for analysis
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
        
        crop_roi = img_np[y1:y2, x1:x2]
        if crop_roi.size == 0: continue
        
        gray = cv2.cvtColor(crop_roi, cv2.COLOR_RGB2GRAY) if len(crop_roi.shape) == 3 else crop_roi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if cls_id in picture_classes:
            if analyze_content_type(binary) == 'text':
                target_cls = 10 

        # Note: target_cls might have changed to 10 above
        if target_cls in text_classes or target_cls == 10:
            # Dynamic snapping to capture Khmer diacritics correctly
            refined_box, is_ignored = snap_to_ink(img_np, box, padding=Config.PADDING, lookahead=15)
        else:
            # Solid validation for real pictures/tables
            refined_box = validate_non_text_content(img_np, box)
            if refined_box is not None:
                refined_box = [
                    max(0, refined_box[0]-Config.PADDING), max(0, refined_box[1]-Config.PADDING),
                    min(img_np.shape[1], refined_box[2]+Config.PADDING), min(img_np.shape[0], refined_box[3]+Config.PADDING)
                ]

        if refined_box is not None:
            raw_results.append((refined_box, target_cls))

    # Apply collision suppression to remove stacked boxes
    filtered_results = suppress_overlapping_boxes(raw_results, overlap_threshold=0.7)

    # Final lists
    for box, cls_id in filtered_results:
        output_boxes.append((box, cls_id))
        output_crops.append(img.crop(box))

    return output_crops, output_boxes