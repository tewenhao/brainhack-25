# Dockerfile for building the ASR image.

# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
# FROM python:3.12-slim
# FROM nvcr.io/nvidia/pytorch:25.03-py3

######### for parakeet
# FROM python:3.10.16-slim
# FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
RUN apt-get update && \
    apt-get -y install gcc && \
    apt-get install build-essential -y && \
    apt-get install -y ffmpeg
#########

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src .
COPY model-v3 ./model

# Starts your model server.
CMD uvicorn asr_server:app --port 5001 --host 0.0.0.0
