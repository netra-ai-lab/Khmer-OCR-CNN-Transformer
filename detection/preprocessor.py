import torch
from PIL import Image
from transformers import SegformerImageProcessor
from detection.config import Config

class Preprocessor:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained(Config.PROCESSOR_ID)
        
        self.processor.size = {"height": Config.IMAGE_SIZE, "width": Config.IMAGE_SIZE}

    def prepare_image(self, image_path):
        """
        Loads image, applies SegFormer normalization/resizing, 
        and returns data for both model and post-processing.
        """

        image = Image.open(image_path).convert("RGB")
        original_size = image.size 
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        pixel_values = inputs["pixel_values"]
        
        return image, pixel_values, original_size