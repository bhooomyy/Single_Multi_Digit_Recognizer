#!/bin/bash

echo "🔥 Starting Training Pipeline..."

# Activate virtual environment if running locally
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "📌 Training Single-Digit Model..."
python src/train_single.py \
    --data_root train \
    --epochs 5 \
    --batch_size 64 \
    --lr 1e-3

echo "📌 Training Multi-Digit Sequence Model..."
python src/train_sequence.py \
    --train_root train \
    --test_root test \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3 \
    --max_len 5 \
    --digit_size 32

echo "🎉 Training Completed Successfully!"
