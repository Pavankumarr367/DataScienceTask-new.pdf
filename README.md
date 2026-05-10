
# Face Authentication API

## Overview
This project implements Face Verification using FastAPI and DeepFace.

Features:
- Accepts two face images
- Detects faces
- Extracts embeddings
- Computes similarity
- Returns:
  - same/different person
  - similarity score
  - bounding boxes

## Files

- train.py -> model configuration
- test.py -> prediction API
- requirements.txt -> dependencies

## Run

Install packages:

pip install -r requirements.txt

Run training:

python train.py

Run API:

uvicorn test:app --host 127.0.0.1 --port 8002

Open:

http://127.0.0.1:8002/docs
