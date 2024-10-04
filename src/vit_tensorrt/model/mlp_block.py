from torch import nn

from vit_tensorrt.config import MLPConfig


class MLPBlock(nn.Sequential):
    def __init__(self, input_size: int, dropout: float, config: MLPConfig):
        layers: list[nn.Module] = []

        # Construct the layers of the MLP
        # This ensures that the output shape is equal to the input shape
        current_input_size = input_size
        for size in config.hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, size, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_input_size = size
        layers.append(nn.Linear(current_input_size, input_size))
        layers.append(nn.Dropout(dropout))

        super().__init__(*layers)

        # Ensure the layers are initialised in the desired fashion
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
