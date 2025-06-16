# brainhack-25

import torch's til-ai 2025 repository!

## 🗣️ Automatic Speech Recognition (ASR)

This year's ASR task was noticeably tougher than last year's. Right from the start, we could tell the audio samples were significantly noisier — a stark contrast to the cleaner recordings from previous iterations.

### 🎧 Whisper: Our Starting Point

Our initial approach was to fine-tune `whisper-medium` on the provided dataset, following the strategy that served us well last year. However, it quickly became apparent that things wouldn’t be so straightforward. When tested on the local data, the model performed abysmally — transcribing in _Malay_ instead of English. An odd but intriguing outcome, possibly due to the heightened noise levels.

To ensure English output, we switched to the `whisper-small.en` model and fine-tuned it using the same dataset. The performance improved, achieving a score of **0.820** on the organisers’ hidden test set — a step up, but still not ideal.

As a sanity check, we evaluated the pretrained `whisper-small.en` model without fine-tuning. It yielded a score of **0.768**, confirming that our fine-tuning helped — just not enough to reach the performance we were aiming for.

We also considered denoising the audio before inference using [`demucs`](https://github.com/facebookresearch/demucs), hoping it would help Whisper handle the noisy inputs better. However, setting up Demucs was a pain, and the output quality didn’t improve much. Given the tight timeframe during qualifiers, we ultimately decided it wasn’t worth diving further down the rabbit hole for such minimal returns.

### 🦜 Pivoting to Parakeet

Given the limitations of Whisper this year, we decided to pivot to NVIDIA’s NeMo Parakeet ASR models. This turned out to be a much stronger direction.

Using the pretrained `parakeet-rnnt-0.6b`, we achieved **0.863** accuracy on the hidden test set. With `parakeet-rnnt-1.1b`, the accuracy jumped to **0.895**. While the smaller model (`0.6b`) was noticeably faster — an advantage in a setting where inference speed was scored — the scoring weightage still gave the edge to the more accurate `1.1b` model.

We attempted to fine-tune `parakeet-rnnt-1.1b` as well, but ran into significant roadblocks. Despite following NVIDIA’s finetuning guide closely and preparing the dataset in the expected format, we encountered persistent CUDA Graph errors when training on Vertex AI. Other teams reported similar problems and got around them by training locally — an option we didn’t have access to.

As a result, our repository still includes the training scripts (which should work), but we ultimately submitted the pretrained `parakeet-rnnt-1.1b` model for both the qualifiers and finals. This is also why model weights are not provided by us in this repository. Please obtain them from [Huggingface](https://huggingface.co/nvidia/parakeet-rnnt-1.1b).

### 🏁 Final Thoughts

Interestingly, on finals day, the pretrained `parakeet-rnnt-1.1b` model held up quite well. But like last year, ASR accuracy wasn’t the decisive factor — the Reinforcement Learning component played a much larger role in the semi-finals and finals. While we’re proud of our ASR work and the decision to pivot mid-way, we recognise that it was just one piece of the overall puzzle in securing our 2nd place finish.

## Computer Vision (CV)

In contrast to the previous year's task, the images this year contained a fixed 18 object classes.  A combined object detection and classification model with high accuracy and speed is therefore ideal for a simpler finetuning process and the lack of need to do zero shot classification.

The data was provided in COCO annotation format and conversion to YOLO format was required to use [YOLOv11](https://docs.ultralytics.com/models/yolo11/).

This was done with data_process.ipynb where an augmented function from the official [ultralytics](https://github.com/ultralytics/ultralytics) library was used.

### Finetuning

Ultralytics' Yolo11 trainer function has built in augmentation parameters and the appropriate ones were chosen (Hue, distortion etc.). However the test results on a yolo11 Large model trained to the point before overfitting had lackluster results of around 0.3 mAP50-95 accuracy.

The most likely reason was the test data contained extremely small objects (some as small as a few pixels) of which none appeared in any of the training data. Any augmentation libraries featuring resize would alter the size of the whole image (which would be changed to 400x400 by the trainer regardless).

data_aug.ipynb contains a custom augmentation script to extract the objects from each image via their bounding box, shrink it significantly before pasting back onto original images together with new labels.

This boosted our final accuracy to around 0.75 for the in person stage.
