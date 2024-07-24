import onnx
import torch
import torch.nn as nn
from onnx2pytorch import ConvertModel

# Load the ONNX model
onnx_model_path = 'models/onnx_model.onnx'
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX model to PyTorch model
pytorch_model = ConvertModel(onnx_model)

# Print model's parameters
for name, param in pytorch_model.named_parameters():
    print(f"Parameter name: {name}")
    print(f"Parameter shape: {param.shape}")
    print(f"Parameter data: {param.data}")

# Alternatively, if you want to see the entire model architecture
print(pytorch_model)
