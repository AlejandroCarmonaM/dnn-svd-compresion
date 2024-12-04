import torch
from models.compressed_model import CompressedNN
from utils.compression import compress_layer
import os

# compress.py
def compress_model(original_model, compression_ratio=0.75):
    compressed_layers = []
    
    # Get all linear layers from original model
    linear_layers = [m for m in original_model.features if isinstance(m, torch.nn.Linear)]
    
    # Compress each linear layer
    for i, layer in enumerate(linear_layers):
        # For the last layer (output layer), use fewer singular values
        if i == len(linear_layers) - 1:
            k = min(10, int(min(layer.weight.shape) * compression_ratio))
        else:
            k = int(min(layer.weight.shape) * compression_ratio)
            
        first_layer, second_layer = compress_layer(layer, k)
        compressed_layers.extend([first_layer, second_layer])
    
    # Create new compressed model
    compressed_model = CompressedNN(compressed_layers)
    
    return compressed_model