# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from models.base_model import ImprovedNN
import os

def train_model():
    os.makedirs('models', exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    
    valset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    
    best_acc = 0
    
    for epoch in range(15):
        model.train()
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in valloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = correct / total
        print(f'Epoch {epoch+1}: Validation Accuracy = {acc:.4f}')
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/original_model.pth')
    
    return model

if __name__ == "__main__":
    train_model()
    print("Training complete. Model saved as models/original_model.pth")