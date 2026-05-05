#!/usr/bin/env bash
set -e
python scripts/make_demo_data.py --config configs/demo.yaml --train_samples 12 --test_samples 4
python scripts/train.py --config configs/demo.yaml
python scripts/infer.py --config configs/demo.yaml --checkpoint outputs/demo/checkpoints/best.pt --num_samples 2 --height 64 --width 64 --no_refine
python scripts/evaluate.py --config configs/demo.yaml --pred_dir outputs/demo/predictions
