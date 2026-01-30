# utils.py
import cv2
import numpy as np
from PIL import Image
from config import Config

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

def snap_to_ink(img_np, box, padding=2, min_ink_pixels=3):
    """
    Refined ink snapping:
    - Lowered min_ink_pixels to 3 to catch small characters.
    - Added a 'safety' check for pixel intensity.
    """
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)

    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0: return None, False

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop

    # Check empty bbox: if the whole crop is very bright, it's definitely empty
    if np.mean(gray) > 250:
        return None, False

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if is_graphical_line(binary):
        return None, True

    coords = cv2.findNonZero(binary)

    if coords is None or len(coords) < min_ink_pixels:
        return None, False

    bx, by, bw, bh = cv2.boundingRect(coords)

    # Filter only if extremely microscopic (less than 2px)
    if bw < 2 or bh < 2:
        return None, False

    new_x1 = max(0, x1 + bx - padding)
    new_y1 = max(0, y1 + by - padding)
    new_x2 = min(img_np.shape[1], x1 + bx + bw + padding)
    new_y2 = min(img_np.shape[0], y1 + by + bh + padding)

    return [new_x1, new_y1, new_x2, new_y2], False

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
        if h < 2 or w < 2: continue

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

    output_boxes = []
    output_crops = []

    for (box, cls_id) in intermediate_results:

        if cls_id in Config.TEXT_CLASSES:

            refined_box, is_line = snap_to_ink(img_np, box, padding=Config.PADDING)
            if refined_box is None: continue
        else:
            p = Config.PADDING
            refined_box = [
                max(0, box[0]-p), max(0, box[1]-p),
                min(img_np.shape[1], box[2]+p), min(img_np.shape[0], box[3]+p)
            ]

        output_boxes.append((refined_box, cls_id))
        output_crops.append(img.crop(refined_box))

    return output_crops, output_boxes