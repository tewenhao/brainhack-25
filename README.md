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

## Optical Character Recognition (OCR)

For this year's TIL-AI, the OCR task was added, which was to produce a transcription given an image of a scanned document. We were provided with JPEG image files, text files with the actual text in each image, and a [hOCR](https://en.wikipedia.org/wiki/HOCR) that included word-, line-, and paragraph-level bounding boxes to train our OCR model.

### Tesseract

For the qualifiers, we decided to use Google's Tesseract OCR engine, specifically [`pytesseract`](https://github.com/h/pytesseract), a wrapper of the OCR engine for Python. When we ran the pre-trained model, the evaluation score was not great, with an accuracy score of **0.779** and a speed score of **0.268**. It was then we realized that `pytesseract` does not run on GPU, which means that we would be limiting our OCR performance if we continued using our CPU bound model. Thus, we knew that we had to look into other alternatives that can offer better performance, as well as the option for fine-tuning.

### EasyOCR

One option that we pursued was [EasyOCR](https://github.com/JaidedAI/EasyOCR), which offered better performance on noisy images and the ability to fine-tune. However, when we ran the pre-trained model, it took way too long, with a shocking speed score of **0.000**. We were less than thrilled, but we still had one thing we could try: fine-tuning it with our training data set.

To fine-tune our EasyOCR model, we followed a guide by Eivind Kjosbakken [here](https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/). It involves using the [`deep-text-recognition-benchmark`](https://github.com/clovaai/deep-text-recognition-benchmark})repository to convert our images to the Lightning Memory-Mapped Database Manager (LMDB) format for training. After performing data pre-processing as outlined in the aforementioned guide, we tried running the training script, but alas to no avail.

It is important to note that while the guide above is generally useful, there are a lot of missing information that can make the process confusing. One of which is how to use your custom fine-tuned model with the `easyocr` Python module. Steps are outlined in the EasyOCR repository to do so, but still it is lacking a lot of details, such as how to setup the model configuration file, which requires information about the model architecture etc.

Despite our best efforts to fine-tune it, EasyOCR was not a good fit for our OCR task.

## Reinforcement Learning (RL)

## Surprise Challenge

The surprise challenge involved reconstructing a shredded document from a set of vertical image strips. Each strip represented a vertical slice of the original image, and our goal was to determine the correct ordering of the strips to reassemble the document. We were told to assume that all slices were upright and of equal dimensions.

We modelled this problem as a Travelling Salesperson Problem (TSP). By defining a pairwise similarity score between every two slices—based on how well the right edge of one matched the left edge of another—we constructed a similarity matrix. The task then reduced to finding a Hamiltonian path through the strips that minimised the total "dissimilarity" score.

### Edge Similarity Metrics

We experimented with several edge similarity metrics:

- Mean Squared Error (MSE) between edge pixels
- Cosine similarity of edge vectors
- Sobel-based gradient similarity
- Structural Similarity Index Measure (SSIM)

Ultimately, we chose MSE due to its simplicity, ease of implementation, and surprisingly competitive performance. Our key insight was that performance improved significantly when we reduced the edge comparison width, likely because a narrower edge minimised noise from image content and focused comparison on direct pixel transitions.

### TSP Strategy

To solve the TSP, we implemented a greedy nearest-neighbour heuristic. For robustness, we ran this heuristic starting from every possible node (i.e., treating each strip as a potential leftmost edge) and selected the route with the lowest overall dissimilarity score. While we recognise that dynamic programming or more advanced solvers (like OR-Tools or Held-Karp) could yield more optimal results or better performance, we opted for the simpler greedy approach due to the limited time available (only one day).

### Reflections

This challenge highlighted how effective simple heuristics and basic image processing techniques can be when paired with the right formulation. While our implementation could certainly be optimised further—particularly in terms of TSP solving—we're pleased with the performance improvements achieved through careful metric selection and parameter tuning.
