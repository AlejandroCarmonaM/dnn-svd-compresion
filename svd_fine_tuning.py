# Apply SVD compression to the model
import torch
import torch.nn as nn
from models.base_model import ImprovedNN
import time
# Prepare data loaders
from torchvision import datasets, transforms

# validate_models.py
def validate_model(model, valloader, model_name=""):
    correct_count, all_count = 0, 0
    start_time = time.time()
    
    model.eval()
    with torch.no_grad():
        for images, labels in valloader:
            # Process full batches instead of individual images
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct_count += predicted.eq(labels).sum().item()
            all_count += labels.size(0)
    
    end_time = time.time()
    inference_time = end_time - start_time
    accuracy = correct_count/all_count
    
    print(f"{model_name} Results:")
    print(f"Number Of Images Tested = {all_count}")
    print(f"Model Accuracy = {accuracy:.4f}")
    print(f"Inference Time = {inference_time:.2f} seconds")
    
    return accuracy, inference_time

def svd_compress(layer, rank):
    # Get weight matrix W and transpose it to match mathematical notation
    # W.t() shape: (in_features, out_features)
    W_t = layer.weight.data.t()
    
    # Perform SVD on W.t(): W.t() = U * S * V^T
    # U: (in_features, in_features)
    # S: (min(in_features,out_features),)
    # V: (out_features, out_features)
    U, S, V = torch.svd(W_t)
    
    # Keep top 'rank' components
    U_r = U[:, :rank]          # Shape: (in_features, rank)
    S_r = S[:rank]             # Shape: (rank,)
    V_r = V[:, :rank]          # Shape: (out_features, rank)
    S_r_diag = torch.diag(S_r) # Shape: (rank, rank)
    
    # Prepare weight matrices for the two new layers
    # For W = (W.t()).t() = (U * S * V^T).t() = V * S * U^T
    
    # First layer: nn.Linear(in_features, rank)
    W1 = U_r               # Shape: (rank, in_features)
    
    # Second layer: nn.Linear(rank, out_features)
    W2 = S_r_diag @ V_r.t()    # Shape: (rank, out_features)
    
    # Create the compressed layers
    first_layer = nn.Linear(layer.in_features, rank, bias=False)
    second_layer = nn.Linear(rank, layer.out_features, bias=True)
    
    # Assign weights and bias
    # Note: We need to transpose the weight matrices back to match the original layer shape
    first_layer.weight.data = W1.t()
    second_layer.weight.data = W2.t()
    second_layer.bias.data = layer.bias.data.clone()
    
    return nn.Sequential(first_layer, second_layer)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

####################################################################################################
####################### MAIN FUNCTION #############################################################
####################################################################################################

# Help Message : python3 svd_fine_tuning.py -model_path [path_to_model.pth] -rank [rank] -epochs [epochs]
import argparse

parser = argparse.ArgumentParser(
    description='SVD Compression and Fine-Tuning',
    usage='python3 svd_fine_tuning.py -model_path [path_to_model.pth] -rank [rank] -epochs [epochs]'
)
parser.add_argument('-model_path', type=str, default='models/original_model.pth', help='Path to the pre-trained model (default = models/original_model.pth)')
parser.add_argument('-rank', type=int, default=20, help='Rank for SVD compression (default = 20)')
parser.add_argument('-epochs', type=int, default=5, help='Number of epochs for fine-tuning (default = 5)')
args = parser.parse_args()

# Load your pre-trained model
print("##############################################\n ORIGINAL MODEL \n##############################################")
model = ImprovedNN()
model.load_state_dict(torch.load(args.model_path))
model.eval()
# PRINT model parameter count
original_model_size = count_parameters(model)
print(f"Original model has {original_model_size} parameters.")

# validate the original model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

valset = datasets.MNIST('./data', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
original_acc, original_time = validate_model(model, valloader, model_name="Original Model")

print("\n##############################################\n COMPRESSED MODEL \n##############################################")

print("Compressing the model...")

# Compress each linear layer in the model
for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        compressed_layer = svd_compress(layer, args.rank)  # Adjust 'rank' based on desired compression
        # Navigate to the parent module
        modules = name.split('.')
        parent_module = model
        for mod in modules[:-1]:
            parent_module = getattr(parent_module, mod)
        # Replace the layer in the parent module
        setattr(parent_module, modules[-1], compressed_layer)

# PRINT model parameter count after compression
compressed_model_size = count_parameters(model)
print(f"Compressed model has {compressed_model_size} parameters.")

# Save the compressed model
torch.save(model.state_dict(), 'models/compressed.pth')
print("Model saved successfully after compression.")

# validate the compressed model
compressed_acc, compressed_time = validate_model(model, valloader, model_name="Compressed Model")

print("\n##############################################\n COMPRESSED AND FINE TUNED MODEL \n##############################################")


# Fine-tune the compressed model
print("Fine-tuning the model (Retraining)...")
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Training loop
model.train()
for epoch in range(args.epochs):  # Adjust the number of epochs as needed
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# PRINT model parameter count after fine-tuning
fine_tuned_model_size = count_parameters(model)
print(f"Fine-tuned model has {fine_tuned_model_size} parameters.")

# Save the fine-tuned model
torch.save(model.state_dict(), 'models/compressed_and_finetuned.pth')
print("Model saved successfully after fine-tuning.")

# validate the compressed and fine-tuned model
valset = datasets.MNIST('./data', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

model.load_state_dict(torch.load('models/compressed_and_finetuned.pth'))
model.eval()

fine_tuned_acc, fine_tuned_time = validate_model(model, valloader, model_name="Compressed and Fine-Tuned Model")

# COMPLETE RESULTS 
print("\n##############################################\n RESULTS \n##############################################")
# Model sizes
print("\nModel Sizes:")
print(f"Original model size: {original_model_size} parameters")
print(f"Compressed model size: {compressed_model_size} parameters, Compression ratio: {compressed_model_size/original_model_size:.2f}")
print(f"Fine-tuned model size: {fine_tuned_model_size} parameters, Compression ratio: {fine_tuned_model_size/original_model_size:.2f}")

# Validation results
print("\nAccuracy Results:")
print(f"Original Model: Accuracy = {original_acc:.4f}")
print(f"Compressed Model: Accuracy = {compressed_acc:.4f}, Precision loss = {original_acc - compressed_acc:.4f}")
print(f"Fine-tuned Model: Accuracy = {fine_tuned_acc:.4f}, Precision loss = {original_acc - fine_tuned_acc:.4f}")

# Inference time
print("\nInference Time:")
print(f"Original Model: {original_time:.2f} seconds")
print(f"Compressed Model: {compressed_time:.2f} seconds, Speedup = {original_time/compressed_time:.2f}")
print(f"Fine-tuned Model: {fine_tuned_time:.2f} seconds, Speedup = {original_time/fine_tuned_time:.2f}")

print("\nAll done!")

