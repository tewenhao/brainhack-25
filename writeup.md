# Writeups

Here, we detail our thought process (and pains) behind each task. Upon reflection, perhaps the stars really did align for us to get to where we eventually did get to. After the fumble in 2024, 2025 is, perhaps, our year.

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

For this year's TIL-AI, an OCR task was introduced. The objective was to produce accurate transcriptions from scanned document images. We were provided JPEG image files, corresponding text files, and [hOCR](https://en.wikipedia.org/wiki/HOCR) annotations containing word-, line-, and paragraph-level bounding boxes to train our OCR model.

### Tesseract

For the qualifiers, we started with Google’s Tesseract OCR engine, using the [`pytesseract`](https://github.com/h/pytesseract) Python wrapper. While easy to set up and use, its performance was underwhelming: an accuracy score of **0.779** and a speed score of **0.268**. We soon realised a key limitation—Tesseract runs on the CPU only. Given how much time inference took, we knew we had to move to GPU-compatible models that allowed fine-tuning.

### EasyOCR

Next, we explored [EasyOCR](https://github.com/JaidedAI/EasyOCR), which showed more promise—on paper. It performed better on noisy images and offered fine-tuning capabilities. Unfortunately, the pre-trained model was extremely slow, yielding a speed score of **0.000**.

Hoping to improve performance, we attempted to fine-tune EasyOCR using our dataset. Following [a guide by Eivind Kjosbakken](https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/), we preprocessed our data and converted it into LMDB format using the [`deep-text-recognition-benchmark`](https://github.com/clovaai/deep-text-recognition-benchmark) repository. However, running the training scripts proved fruitless.

While the guide was a helpful starting point, it lacked clarity on crucial integration steps—especially when trying to load the fine-tuned model back into the `easyocr` Python module. Despite additional documentation in the EasyOCR repo, missing details such as how to correctly configure the model architecture made it frustrating to work with. Ultimately, EasyOCR didn’t make the cut.

### PaddleOCR

We also considered PaddleOCR, which showed initial promise. However, integrating it into our setup turned out to be a major blocker. We eventually discovered—after the submission deadline and some helpful conversations with other teams—that our Python version was incompatible and that Paddle had to be downgraded to version `2.5.2` for CPU inference to work.

As for GPU inference, Ada from the tech team kindly informed us that we’d need to build PaddleOCR from source with `CUDA 12.8`. Unfortunately, given our time constraints and other priorities, we were unable to overcome this setup hurdle, and had to abandon PaddleOCR as a viable option.

### SmolDocling

We also explored the use of Visual Language Models (VLMs), which have shown strong performance on OCR-style tasks. One model we tested was [`SmolDocling`](https://huggingface.co/ds4sd/SmolDocling-256M-preview).

In our early tests, SmolDocling produced excellent results when run on individual images. However, it failed to scale when evaluated on the hidden test set—timing out repeatedly. The root cause appeared to be its reliance on [`flash attention`](https://github.com/Dao-AILab/flash-attention), an optimised attention mechanism that wasn't supported by our environment. Installing it via `pip` was also extremely slow, making it impractical to use under competition constraints.

## Reinforcement Learning (RL)

Model training failed.

So we made an algo that came out of desperation.

For scout it calculates the value of every cell in the map. The value of each cell is determined by the value of the points on it and cells that are further from the Scout, it is also affected by the presence of enemies nearby the cell.

Because it was a last minute solution, it's calculation are nowhere as precise as the first place algorithm.

For guards it just heads for set positions around the map and chases the Scout if the guard sees it. It only stops once it is near the last seen position of the scout and cannot see the scout.

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
