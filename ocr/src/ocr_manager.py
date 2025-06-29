"""Manages the OCR model."""
import pytesseract
from PIL import Image
import io


class OCRManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        pass

    def ocr(self, image: bytes) -> str:
        """Performs OCR on an image of a document.

        Args:
            image: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """

        # Your inference code goes here.
        
        return pytesseract.image_to_string(Image.open(io.BytesIO(image)))