#!/usr/bin/env python3
"""Minimal test to load the model."""

import sys
from pathlib import Path

import torch
print(f"PyTorch version: {torch.__version__}")

# Add transformer to path
sys.path.append(str(Path(__file__).parent.parent / "transformer"))

print("About to import TransformerRegressor...")
from transformer_functions import TransformerRegressor
print("Import successful!")

print("Loading checkpoint...")
checkpoint = torch.load('../models/model_4pt.pt', map_location='cpu')
print(f"Checkpoint loaded. Keys: {list(checkpoint.keys())}")

print("Creating model...")
model_args = checkpoint['model_args'].copy()
model_args['device'] = 'cpu'
print(f"Model args: {model_args}")

model = TransformerRegressor(**model_args)
print("Model created!")

print("Loading weights...")
model.load_state_dict(checkpoint['model_state_dict'])
print("Weights loaded!")

model.eval()
print("Model ready for inference!")
