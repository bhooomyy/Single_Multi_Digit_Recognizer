# 🧠 Multi-Digit Handwritten Sequence Recognizer (PyTorch)

This project implements an end-to-end deep learning system for recognizing:
- **Single handwritten digits (0–9)**
- **Multi-digit numerical sequences (1–5 digits)** without explicit digit segmentation

The system uses a **CNN backbone + sliding-window feature extraction + Conv1D sequence head** to decode full digit strings efficiently.

---

## 📌 Overview

### ✔ Single-Digit Recognition  
A convolutional neural network trained on MNIST-style digits.

### ✔ Multi-Digit Sequence Recognition  
The model recognizes whole sequences (e.g., `407`, `92`, `50123`) by:
1. Padding each sequence image to a fixed width  
2. Extracting **32×32 sliding windows**  
3. Embedding each window using a **shared CNN backbone**  
4. Passing embeddings through a **Conv1D decoder**  
5. Predicting digits position-wise  
6. Using a **blank class** to support sequences shorter than 5 digits  

---

## ✨ Features

- Fully implemented in **PyTorch**
- Handles sequences up to **5 digits**
- No manual segmentation  
- Shared CNN backbone + Conv1D head  
- Supports **variable-length outputs with blank padding**
- Tracks:
  - Per-digit accuracy
  - Full sequence accuracy
- Includes:
  - Dataset loaders  
  - Training scripts  
  - Evaluation utilities  
  - Visualization tools  
  - Model summaries  

---

## 📂 Dataset

### 1) Single-digit dataset  
MNIST / Kaggle Digit Recognizer style (28×28 grayscale digits).

### 2) Multi-digit dataset (synthetic)
Images created by horizontally concatenating 1–5 MNIST digits.  
Each digit resized to **32×32**.

File labels come from filenames.  
Example:
