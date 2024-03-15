import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, channels_in, channels_out, activation=None):
        super().__init__()
        
        layers = [nn.Linear(channels_in, channels_out, bias=True)]
        self.activation = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
        }.get(activation, None)
        
        if activation is not None:
            layers.append(self.activation)
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
    def _init_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    
class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
class CosActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class FourierBlock(nn.Module):
    def __init__(self, channels_in, channels_out, activation=None):
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
        return self.layers_sin(x) + self.layers_cos(x)
    
    def _init_weights(self):
        for layers in [self.layers_sin, self.layers_cos]:
            for layer in layers:
                if hasattr(layer, 'weight') and layer.weight is not None:
                    nn.init.xavier_uniform_(layer.weight)

    
class FCNet(nn.Module):
    def __init__(self, input_features, hidden_layers, output_features, activation, init_weights=False):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_layers = hidden_layers
        self.output_features = output_features
        self.activation = activation

        blocks = self._create_blocks()
        self.blocks = nn.Sequential(*blocks)
        
        if init_weights:
            self._init_weights()
            
        self.param_count = self._calculate_params()
    
    def forward(self, x):
        return self.blocks(x)
    
    def _calculate_params(self):
        return sum(p.numel() for p in self.parameters())

    def _create_blocks(self):
        
        block_class = FourierBlock if self.activation=='cas' else LinearBlock
        blocks = [block_class(self.input_features, self.hidden_layers[0], self.activation)]
        if len(self.hidden_layers) > 1:
            for i in range(len(self.hidden_layers) - 1):
                blocks.append(block_class(self.hidden_layers[i], self.hidden_layers[i+1], self.activation))
        blocks.append(LinearBlock(self.hidden_layers[-1], self.output_features, activation=None))
        
        return blocks
    
    def _init_weights(self):
        for block in self.blocks:
            block._init_weights()