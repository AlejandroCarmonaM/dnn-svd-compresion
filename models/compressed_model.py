import torch.nn as nn

class CompressedNN(nn.Module):
    def __init__(self, compressed_layers):
        super(CompressedNN, self).__init__()
        self.layers = nn.ModuleList(compressed_layers)
        
        # Add BatchNorm layers for each compressed layer pair output
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(compressed_layers[i+1].out_features) 
            for i in range(0, len(compressed_layers)-2, 2)
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 784)
        
        # Process through compressed layers with BatchNorm and Dropout
        for i in range(0, len(self.layers)-2, 2):
            # Compressed layer pair
            x = self.layers[i](x)
            x = self.layers[i+1](x)
            
            # Add BatchNorm, ReLU, and Dropout for hidden layers
            x = self.batch_norms[i//2](x)
            x = self.relu(x)
            x = self.dropout(x)
            
        # Final layer pair (output layer)
        x = self.layers[-2](x)
        x = self.layers[-1](x)
        
        return x