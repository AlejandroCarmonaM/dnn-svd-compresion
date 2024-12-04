import torch
import torchvision
from torchvision import datasets, transforms
from models.base_model import ImprovedNN
from models.compressed_model import CompressedNN
import time
from compress import compress_model

def load_models():
    # Load original model
    original_model = ImprovedNN()
    original_model.load_state_dict(torch.load('models/original_model.pth'))
    original_model.eval()
    
    # In validate_models.py
    compression_ratio = 0.15  # Keep 50% of singular values
    compressed_model = compress_model(original_model, compression_ratio)
    
    # Save and verify compressed model has layers
    if len(compressed_model.layers) == 0:
        raise ValueError("Compressed model has no layers!")
        
    torch.save(compressed_model.state_dict(), 'models/compressed_model.pth')
    print(f"Compressed model saved successfully with {len(compressed_model.layers)} layers")
    
    compressed_model.eval()
    return original_model, compressed_model

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
    
    print(f"\n{model_name} Results:")
    print(f"Number Of Images Tested = {all_count}")
    print(f"Model Accuracy = {accuracy:.4f}")
    print(f"Inference Time = {inference_time:.2f} seconds")
    
    return accuracy, inference_time

# Calculate model sizes
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Prepare validation data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    valset = datasets.MNIST('./data', download=True, train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    
    # Load both models
    original_model, compressed_model = load_models()
    
    original_size = count_parameters(original_model)
    compressed_size = count_parameters(compressed_model)

    print(f"Original model size: {original_size} parameters")
    print(f"Compressed model size: {compressed_size} parameters")
    print(f"Compression ratio: {compressed_size/original_size:.2f}")
    
    # Validate both models
    orig_acc, orig_time = validate_model(original_model, valloader, "Original Model")
    comp_acc, comp_time = validate_model(compressed_model, valloader, "Compressed Model")
    
    # Compare results
    print("\nComparison:")
    print(f"Accuracy difference: {(orig_acc - comp_acc)*100:.2f}%")
    print(f"Speed improvement: {(orig_time - comp_time)/orig_time*100:.2f}%")

if __name__ == "__main__":
    main()