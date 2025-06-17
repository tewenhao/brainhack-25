# Dockerfile for building the OCR image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
# FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310
# FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# to fix cv2 error: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# solution from: https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

Installs Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copies local Paddlex models and fonts
# COPY ./PaddleOCR/official_models /root/.paddlex/official_models
# COPY ./PaddleOCR/fonts /root/.paddlex/fonts

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# RUN python -c "\
# import easyocr; \
# reader = easyocr.Reader(['en']);"


# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn ocr_server:app --port 5003 --host 0.0.0.0
