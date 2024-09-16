from collections import OrderedDict

import torch
from torch.nn import (
    Conv2d,
    Dropout,
    init,
    GELU,
    LayerNorm,
    Linear,
    Module,
    MultiheadAttention,
    Parameter,
    Sequential,
)

from vit_tensorrt.utils import MetaLogger


class ViT(Module, MetaLogger):
    def __init__(
        self,
        image_size: int = 768,
        patch_size: int = 16,
        patch_embedding_size: int = 768,
        num_heads: int = 16,
        mlp_hidden_size: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_encoder_layers: int = 24,
        device: str = "cuda:0",
    ):
        Module.__init__(self)
        MetaLogger.__init__(self)

        self._set_compute_device(device)

        # Convolves over the image extracting patches of size `patch_size * patch_size`.
        # This ensures the output embedding contains `patch_embedding_size` channels.
        self.patcher = Conv2d(3, patch_embedding_size, patch_size, patch_size)

        # Determine the number of embeddings after the conv layer
        sequence_length = (image_size // patch_size) ** 2

        # Create a learnable [class] token
        # We use the class token because encoders are squence-to-squence models, so the
        # input sequence length equals the output sequence length.
        #   - If we applied the linear layer to all sequence outputs it would fix the
        #     input image size.
        #   - If we applied it to a single element of the sequence, our output my be
        #     unfairly biased towards the corresponding patch.
        # By using the class token, we can use the output corresponding to it as the
        # input to the linear layer.
        self.class_token = Parameter(torch.zeros(1, 1, patch_embedding_size))
        sequence_length += 1

        self.encoder = Encoder(
            sequence_length,
            patch_embedding_size,
            num_heads,
            mlp_hidden_size,
            dropout,
            attention_dropout,
            num_encoder_layers,
        )

    def _set_compute_device(self, device: str):
        if "cuda" in device and torch.cuda.is_available():
            torch.set_default_device(device)
        elif "cuda" in device and not torch.cuda.is_available():
            self.logger.warning("CUDA device not found - running on CPU.")
            torch.set_default_device("cpu")
        else:
            torch.set_default_device(device)

        self.logger.info(f"Set default device to {torch.get_default_device()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass of the model.

        Parameters
        ----------
        x: `torch.Tensor`
            An input batch of images shaped as `(batch_size, channel, height, width)`.
        """
        grid_embeddings: torch.Tensor = self.patcher(x)
        return grid_embeddings.flatten(2, 3)


class Encoder(Module):
    def __init__(
        self,
        sequence_length: int,
        patch_embedding_size: int,
        num_heads: int,
        mlp_hidden_size: int,
        dropout: float,
        attention_dropout: float,
        num_layers: int,
    ):
        super().__init__()

        # Learnable position embeddings. Each of these is added to the patch embedding
        # encode into the embedding the position of the patch.
        self.position_embeddings = Parameter(
            torch.empty(1, sequence_length, patch_embedding_size).normal_(std=0.02)
        )

        # Create the encoder layers
        layers: OrderedDict[str, Module] = OrderedDict()
        for layer_index in range(num_layers):
            layers[f"encoder_layer_{layer_index}"] = EncoderBlock(
                num_heads,
                patch_embedding_size,
                mlp_hidden_size,
                dropout,
                attention_dropout,
            )
        self.layers = Sequential(layers)

        self.dropout = Dropout(dropout)
        self.norm_layer = LayerNorm(patch_embedding_size, eps=1e-6)


class EncoderBlock(Module):
    def __init__(
        self,
        num_heads: int,
        patch_embedding_size: int,
        mlp_hidden_size: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()

        # Define the attention block
        #   - The normalisation layer ensures that each embeddings is normalised input a
        #     normal distribution.
        #   - Multi-headed attention correlates relationships between input embeddings
        #   - Dropout provides randomisation during training to improve
        #     generalisability.
        self.norm_layer_1 = LayerNorm(patch_embedding_size, eps=1e-6)
        self.self_attention = MultiheadAttention(
            patch_embedding_size, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = Dropout(dropout)

        # Multi-layer perceptron
        #   - Layer norm to ensure output embeddings fit the standard normal
        #     distribution.
        #   - Multi-layer perceptron is used to extract deeper features/relationships.
        self.norm_layer_2 = LayerNorm(patch_embedding_size, eps=1e-6)
        self.mlp = MLPBlock(patch_embedding_size, mlp_hidden_size, dropout)


class MLPBlock(Sequential):
    def __init__(self, input_dimensions: int, hidden_dimension: int, dropout: float):
        layers = [
            Linear(input_dimensions, hidden_dimension, bias=True),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dimension, input_dimensions, bias=True),
            Dropout(dropout),
        ]

        super().__init__(*layers)

        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias, std=1e-6)


if __name__ == "__main__":
    # # instantiate the model
    model = ViT(16)

    # construct input tensor
    input_tensor = torch.rand((1, 3, 256, 256))

    output = model.forward(input_tensor)

    print(output.shape)
