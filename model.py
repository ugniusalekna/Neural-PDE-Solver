import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, channels_in, channels_out, activation):
        super().__init__()
        layers = [
            nn.Linear(channels_in, channels_out, bias=True),
        ]
        match activation:
            case 'relu':
                layers.append(nn.ReLU())
            case 'gelu':
                layers.append(nn.GELU())
            case 'tanh':
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    
class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
class CosActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class FourierBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        layers_sin = [
            nn.Linear(channels_in, channels_out, bias=False),
            SinActivation(),
        ]
        layers_cos = [
            nn.Linear(channels_in, channels_out, bias=False),
            CosActivation(),
        ]
        
        self.layers_sin = nn.Sequential(*layers_sin)
        self.layers_cos = nn.Sequential(*layers_cos)

    def forward(self, x):
        sin_part = self.layers_sin(x)
        cos_part = self.layers_cos(x)
        return sin_part + cos_part
    
    
class FourierNet(nn.Module):
    def __init__(self, input_features, hidden_layers, output_features):
        super().__init__()
        
        self.activation = 'cas'
        self.hidden = hidden_layers
        self.fc_in = FourierBlock(input_features, hidden_layers[0])

        layers = []
        if len(hidden_layers) > 1:
            for i in range(len(hidden_layers) - 1):
                layers.append(FourierBlock(hidden_layers[i], hidden_layers[i+1]))
        self.layers = nn.Sequential(*layers)
        
        self.fc_out = nn.Linear(hidden_layers[-1], output_features, bias=True)
    
    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        
        return x
    
    
class ClassicNet(nn.Module):
    def __init__(self, input_features, hidden_layers, output_features, activation):
        super().__init__()
        
        self.activation = activation
        self.hidden = hidden_layers
        self.fc_in = LinearBlock(input_features, hidden_layers[0], activation)

        layers = []
        if len(hidden_layers) > 1:
            for i in range(len(hidden_layers) - 1):
                layers.append(LinearBlock(hidden_layers[i], hidden_layers[i+1], activation))
        self.layers = nn.Sequential(*layers)
        
        self.fc_out = nn.Linear(hidden_layers[-1], output_features, bias=True)
    
    def forward(self, x):
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        
        return x