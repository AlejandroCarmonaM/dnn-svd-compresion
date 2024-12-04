# utils/compression.py
import torch

def compress_layer(layer, k=None):
    """
    Compress a linear layer using SVD decomposition.
    Args:
        layer: The linear layer to compress
        k: Number of singular values to keep. If None, determined automatically
    """
    weight = layer.weight.data
    U, S, V = torch.svd(weight)
    
    if k is None:
        # Calculate energy preservation (95% of energy)
        total_energy = torch.sum(S)
        cumsum = torch.cumsum(S, dim=0)
        energy_ratio = cumsum / total_energy
        k = torch.sum(energy_ratio <= 0.95).item()
        k = max(1, min(k, min(weight.shape)))
    
    # Take only the top k singular values and vectors
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = V[:, :k]
    
    # Create the two new layers
    first_layer = torch.nn.Linear(weight.shape[1], k, bias=False)
    second_layer = torch.nn.Linear(k, weight.shape[0], bias=True)
    
    # W = Σ * V^T
    W = torch.mm(torch.diag(S_k), V_k.t())
    first_layer.weight.data = W
    
    # U will be the weight matrix of the second layer
    second_layer.weight.data = U_k
    
    # Copy the bias from the original layer
    if layer.bias is not None:
        second_layer.bias.data = layer.bias.data
    
    print(f"Compressed layer {weight.shape} -> [{weight.shape[1]}, {k}] + [{k}, {weight.shape[0]}]")
    return first_layer, second_layer