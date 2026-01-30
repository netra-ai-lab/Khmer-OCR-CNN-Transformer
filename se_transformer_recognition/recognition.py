import argparse
import sys
import logging
from pathlib import Path

# Import from the source package
from config import OCRConfig
from utils import setup_logging, autodetect_config
from tokenizer import Tokenizer
from predictor import OCRPredictor

# Import your model architecture file
try:
    from model.se_model import KhmerOCR as SE_KhmerOCR
    from model.vgg_model import KhmerOCR as VGG_KhmerOCR
    from model.resnet_model import KhmerOCR as ResNet_KhmerOCR

except ImportError:
    print("Error: 'se_model.py' must be in the same directory.")
    sys.exit(1)

# ==============================================================================
# GLOBAL SETTINGS & STATE
# ==============================================================================
# Define defaults here so you don't have to pass them every time
DEFAULT_MODEL_PATH = "./weight/khmerocr_se_transformer.pth"
DEFAULT_VOCAB_PATH = "char2idx.json"

# Global variable to hold the model in memory (Singleton)
_PREDICTOR_INSTANCE = None

def _get_predictor(model_path=None, vocab_path=None):
    """
    Internal function to load the model only once.
    """
    global _PREDICTOR_INSTANCE
    
    # Use defaults if not provided
    model_path = model_path or DEFAULT_MODEL_PATH
    vocab_path = vocab_path or DEFAULT_VOCAB_PATH

    if "vgg" in model_path.lower():
        model = VGG_KhmerOCR
    elif "resnet" in model_path.lower():
        model = ResNet_KhmerOCR
    else:
        model = SE_KhmerOCR
        
    if _PREDICTOR_INSTANCE is not None:
        return _PREDICTOR_INSTANCE
    try:
        detected_cfg = autodetect_config(model_path)
        config = OCRConfig(**detected_cfg)
        tokenizer = Tokenizer(vocab_path)

        _PREDICTOR_INSTANCE = OCRPredictor(
            model_path=model_path,
            tokenizer=tokenizer,
            config=config,
            model_class=model
        )
        return _PREDICTOR_INSTANCE

    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

# ==============================================================================
# PUBLIC API
# ==============================================================================

def recognize(image_path: str, beam_width: int = 3, model_path=None, vocab_path=None) -> str:
    """
    Recognizes text from an image path.
    
    Args:
        image_path (str): Path to the image file.
        beam_width (int): Beam search width (default 3).
        model_path (str): Optional override for model path.
        vocab_path (str): Optional override for vocab path.
    
    Returns:
        str: The predicted text.
    """
    predictor = _get_predictor(model_path, vocab_path)
    
    try:
        result_text = predictor.predict(image_path, beam_width=beam_width)
        return result_text
    except Exception as e:
        print(f"Prediction error for {image_path}: {e}")
        return ""

# ===================
# CLI ENTRY POINT
# ===================
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Khmer OCR Inference Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to .pth")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH, help="Path to vocab json")
    parser.add_argument("--beam", type=int, default=3, help="Beam width (1 for greedy)")
    parser.add_argument("--output", type=str, help="Save result to text file")
    
    args = parser.parse_args()

    # Call the API function
    text = recognize(args.image, args.beam, args.model, args.vocab)
    
    print("\n" + "="*40)
    print(f"RESULT: {text}")
    print("="*40 + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

    """
    USAGE EXAMPLE:

        python recognition.py --image "test_images/sample.png"

        python recognition.py \
        --image "sample.png" \
        --model "weight/model.pth" \
        --vocab "char2idx.json" \
        --beam 5 \
        --output "results/sample_output.txt"
        
    ARGUMENTS:
        --image: Path to the input image (Required).
        --model: Path to .pth file (Default: ./weight/...).
        --vocab: Path to .json vocab (Default: char2idx.json).
        --beam: Beam width. Set to 1 for Greedy Search (Default: 3).
        --output: (Optional) Text file to save the result.
    
    RUN VIA PYTHON:
        from recognize_line import recognize

        # Basic usage (uses defaults defined in the file)
        text = recognize("test_image_1.png")
        print(text)

        # Processing multiple images (Model stays loaded!)
        images = ["img1.png", "img2.png", "img3.png"]
        for img in images:
            print(f"{img}: {recognize(img)}")

        # Override settings if needed
        text_custom = recognize("test_image.png", beam_width=5, model_path="other_model.pth")
    """