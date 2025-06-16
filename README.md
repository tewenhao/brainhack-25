# brainhack-25
import torch's til-ai 2025 repository!

## Computer Vision

In contrast to the previous year's task, the images this year contained a fixed 18 object classes.  A combined object detection and classification model with high accuracy and speed is therefore ideal for a simpler finetuning process and the lack of need to do zero shot classification.

The data was provided in COCO annotation format and conversion to YOLO format was required to use [YOLOv11](https://docs.ultralytics.com/models/yolo11/).

This was done with data_process.ipynb where an augmented function from the official [ultralytics](https://github.com/ultralytics/ultralytics) library was used.

### Finetuning

Ultralytics' Yolo11 trainer function has built in augmentation parameters and the appropriate ones were chosen (Hue, distortion etc.). However the test results on a yolo11 Large model trained to the point before overfitting had lackluster results of around 0.3 mAP50-95 accuracy. 

The most likely reason was the test data contained extremely small objects (some as small as a few pixels) of which none appeared in any of the training data. Any augmentation libraries featuring resize would alter the size of the whole image (which would be changed to 400x400 by the trainer regardless). 

data_aug.ipynb contains a custom augmentation script to extract the objects from each image via their bounding box, shrink it significantly before pasting back onto original images together with new labels.

This boosted our final accuracy to around 0.75 for the in person stage.

