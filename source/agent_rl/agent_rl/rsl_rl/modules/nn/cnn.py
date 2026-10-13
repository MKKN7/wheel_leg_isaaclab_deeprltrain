import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation

class CustomCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,          # RGB->3
        input_dim=(17,11),      # （H, W）
        output_dim=32,          
        conv_layers=[           # conv param list
            {"out_channels":2, "kernel_size":(9,3), "stride": 1, "padding":0, "pool":None, "batch_norm":True},
            {"out_channels":4, "kernel_size":3, "stride": 1, "padding":0, "pool":None, "batch_norm":True}
        ],
        hidden_layers =[128,64],        # hidden mlp layer
        activation: str = "lrelu",
        output_activation: str = "tanh"
    ):
        super(CustomCNN, self).__init__()
        self.activationStr = activation
        self.activation = resolve_nn_activation(activation)
        output_activation = resolve_nn_activation(output_activation)
        
        # conv list struct
        self.conv_blocks = []
        current_channels = in_channels
        current_size = input_dim
        
        for layer_config in conv_layers:
            # add conv2d
            conv = nn.Conv2d(
                in_channels=current_channels,
                out_channels=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                stride=layer_config["stride"],
                padding=layer_config["padding"]
            )
            self.conv_blocks.append(conv)
            # add batchnorm
            if layer_config["batch_norm"]:
                self.conv_blocks.append(nn.BatchNorm2d(layer_config["out_channels"],affine=False))
            # add activation
            self.conv_blocks.append(self.activation)
            # update dims calculation
            current_channels = layer_config["out_channels"]
            current_size = ((current_size[0]+2*layer_config["padding"]-conv.kernel_size[0])/layer_config["stride"]+1,
                            (current_size[1]+2*layer_config["padding"]-conv.kernel_size[1])/layer_config["stride"]+1)
            
            # add poolings (optional)
            if layer_config["pool"] == "max":
                self.conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_size = (current_size[0] // 2, current_size[1] // 2)
            elif layer_config["pool"] == "avg":
                self.conv_blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
                current_size = (current_size[0] // 2, current_size[1] // 2)
        
        # flatten conv output
        self.flatten_size = int(current_channels * current_size[0] * current_size[1])
        self.conv_blocks.append(nn.Flatten())
        self.conv_blocksL = nn.Sequential(*self.conv_blocks)
        
        # mlp
        self.hidden_layers = []
        prev_units = self.flatten_size
        
        for units in hidden_layers:
            self.hidden_layers.append(nn.Linear(prev_units, units))
            self.hidden_layers.append(self.activation)
            prev_units = units
            
        self.hidden_layersL = nn.Sequential(*self.hidden_layers)
        
        # output layer
        self.output_layer = []
        self.output_layer.append(nn.Linear(prev_units, output_dim))
        self.output_layer.append(output_activation)
        self.output_layerL = nn.Sequential(*self.output_layer)
        
        # initial weights
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_blocksL(x)
        x = self.hidden_layersL(x)
        x = self.output_layerL(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.activationStr == 'relu':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity=self.activationStr)
                elif self.activationStr == 'lrelu':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', a=self.activation.negative_slope)
                else:
                    pass
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., 0.01)
                nn.init.constant_(m.bias, 0)
                
    def reset(self, dones=None):
        pass
    
    def __getitem__(self, idx):
        return self.model[idx]