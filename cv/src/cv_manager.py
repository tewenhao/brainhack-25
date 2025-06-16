"""Manages the CV model."""

import time
from typing import Any
from ultralytics import YOLO
from PIL import Image
import base64
import io
# import torch


class CVManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.model = YOLO('best.pt')
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def base64_to_image(self, base64_string):
        return Image.open(io.BytesIO(base64_string)).convert("RGB")
        

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """
        # time.sleep(10)
        # Your inference code goes here.
        input_bytes = self.base64_to_image(image)
        prediction = []
        
        results = self.model(input_bytes)
        # results = self.model.predict(input_bytes, device=self.device)
        for result in results:
            bboxes = result.boxes
            for box in bboxes:
                # Get (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Convert to (x, y, w, h)
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1 
            
                category_id = int(box.cls[0].item())
            
                prediction.append(
                    {
                        "bbox": [x,y,w,h],
                        "category_id": category_id
                    }
                )

        return prediction
