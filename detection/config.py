import os
import cv2
class Config:
    # Model Setup
    MODEL_PATH = "detection/det-b0" 
    PROCESSOR_ID = "nvidia/mit-b0" 
    NUM_LABELS = 12
    IMAGE_SIZE = 512

    # DocLayNet Label Mapping
    ID2LABEL = {
        0: "Background", 1: "Caption", 2: "Footnote", 3: "Formula", 
        4: "List-item", 5: "Page-footer", 6: "Page-header", 7: "Picture", 
        8: "Section-header", 9: "Table", 10: "Text", 11: "Title"
    }
    
    # Class Color Map (RGB)
    COLORS = {
        1: (255, 255, 0), 2: (0, 255, 255), 3: (255, 0, 255), 4: (0, 128, 128),
        5: (128, 128, 128), 6: (200, 200, 200), 7: (128, 0, 128), 8: (255, 165, 0),
        9: (255, 0, 0), 10: (0, 255, 0), 11: (0, 0, 255)
    }

    # Thresholds
    ENTRY_THRESHOLD = 0.10
    SCORE_THRESHOLD = 0.2
    
    # Morphology & Merging
    MORPH_KERNEL_SIZE = (40, 6)
    MERGE_X_DIST = 60
    LINE_OVERLAP_THRESHOLD = 0.4
    
    # Arbitration
    SIGNIFICANCE_THRESHOLD = 0.10
    
    # Ink Snapping & Filtering
    PADDING = 5
    MIN_INK_PIXELS = 3
    LINE_ASPECT_RATIO = 50
    LINE_DENSITY_THRESHOLD = 0.80
    
    # Categories to apply snapping to
    TEXT_CLASSES = [1, 2, 4, 5, 6, 8, 10, 11]

    OUTPUT_DIR = "detection_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Bounding Box Settings
    DRAW_THICKNESS = 1
    
    # Heatmap Colormap
    HEATMAP_STYLE = cv2.COLORMAP_JET