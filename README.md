# Multi-Digit Handwritten Sequence Recognizer (PyTorch)

This project implements an end-to-end deep learning system for recognizing:
- Single handwritten digits (0–9)
- Multi-digit numerical sequences (1–5 digits) without explicit digit segmentation

The system uses a CNN backbone, sliding-window feature extraction, and a Conv1D sequence head to decode full digit strings efficiently.

---

## Overview

### Single-Digit Recognition
A standard convolutional neural network is trained on MNIST-style digits (28x28 grayscale images).

### Multi-Digit Sequence Recognition
The sequence model recognizes entire digit strings such as:
- 407
- 92
- 50123

Workflow:
1. Pad each sequence image to a fixed width.
2. Extract 32x32 sliding windows.
3. Pass each window through a shared CNN backbone.
4. Collect window embeddings.
5. Decode them using a Conv1D sequence head.
6. Predict digits position-wise (fixed length 5).
7. Use a blank class to support shorter sequences.

---

## Features

- Implemented fully in PyTorch
- Handles 1–5 digit sequences
- No digit segmentation required
- Shared CNN backbone for efficiency
- Conv1D sequence decoder for position predictions
- Supports blank padding for variable-length sequences
- Tracks:
  - Per-digit accuracy
  - Full sequence accuracy

Includes:
- Dataset loaders
- Training scripts
- Evaluation scripts
- Visualization helpers
- Model summary utilities

---

## Dataset

### 1) Single-digit dataset
MNIST / Kaggle Digit Recognizer style grayscale digits.

### 2) Multi-digit synthetic dataset
Created by horizontally concatenating 1–5 MNIST digits.  
Each digit resized to 32x32 before merging.

### Label Format
Labels are extracted from filenames.

Example:
```
4021.png → [4, 0, 2, 1]
```

Shorter sequences are padded with a blank class internally.

### Dataset Link
https://www.kaggle.com/datasets/hojjatk/mnist-dataset (Single Digit)
https://www.kaggle.com/code/fellahabdelnour13/multi-digit-sequence-recognition (Multi-Digit Sequence recognition)
---

## Project Structure (Recommended)

```
.
├── train/                     # Multi-digit training images
├── test/                      # Multi-digit test images
├── src/
│   ├── dataset.py             # Dataset classes
│   ├── model_single.py        # Single-digit CNN
│   ├── model_sequence.py      # CNN + Conv1D sequence model
│   ├── train_single.py        # Training script (single-digit)
│   ├── train_sequence.py      # Training script (multi-digit)
│   ├── eval.py                # Evaluation tools
│   └── utils.py               # Helper functions
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## Training (Examples)

### Train single-digit model
```
python src/train_single.py --data_root <path>
```

### Train multi-digit sequence model
```
python src/train_sequence.py --train_root train --test_root test
```

---

## Evaluation

```
python src/eval.py --test_root test --ckpt outputs/checkpoints/best_sequence_model.pt
```

---

## License
Choose an open-source license (MIT recommended).

---

## Acknowledgements
- MNIST dataset creators
- PyTorch community
- Research on segmentation-free OCR pipelines
