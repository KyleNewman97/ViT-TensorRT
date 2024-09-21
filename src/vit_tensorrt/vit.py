from collections import OrderedDict
from pathlib import Path

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
from vit_tensorrt.vit_config import AttentionConfig, EncoderConfig, MLPConfig, ViTConfig


class ViT(Module, MetaLogger):
    def __init__(
        self,
        config: ViTConfig,
        device: str = "cuda:0",
    ):
        Module.__init__(self)
        MetaLogger.__init__(self)

        self.config = config
        self._set_compute_device(device)

        # Convolves over the image extracting patches of size `patch_size * patch_size`.
        # This ensures the output embedding contains `patch_embedding_size` channels.
        self.patcher = Conv2d(
            3, config.patch_embedding_size, config.patch_size, config.patch_size
        )

        # Determine the number of embeddings after the conv layer
        sequence_length = (config.image_size // config.patch_size) ** 2

        # Create a learnable [class] token
        # We use the class token because encoders are squence-to-squence models, so the
        # input sequence length equals the output sequence length.
        #   - If we applied the linear layer to all sequence outputs it would fix the
        #     input image size.
        #   - If we applied it to a single element of the sequence, our output my be
        #     unfairly biased towards the corresponding patch.
        # By using the class token, we can use the output corresponding to it as the
        # input to the linear layer.
        self.class_token = Parameter(torch.zeros(1, 1, config.patch_embedding_size))
        sequence_length += 1

        self.encoder = Encoder(
            sequence_length,
            config.patch_embedding_size,
            config.encoder_config,
        )

        # Initialise the "head" linear layer
        self.head = Linear(config.patch_embedding_size, config.num_classes)

    def _set_compute_device(self, device: str):
        if "cuda" in device and torch.cuda.is_available():
            torch.set_default_device(device)
        elif "cuda" in device and not torch.cuda.is_available():
            self.logger.warning("CUDA device not found - running on CPU.")
            torch.set_default_device("cpu")
        else:
            torch.set_default_device(device)

        self.logger.info(f"Set default device to {torch.get_default_device()}")

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass of the model.

        Parameters
        ----------
        x: `torch.Tensor`
            An input batch of images shaped as `(batch_size, channel, height, width)`.
        """

        # Ensure the input image has the right shape
        n, _, h, w = image_batch.shape
        if h != self.config.image_size or w != self.config.image_size:
            expected = self.config.image_size
            msg = f"Expected (h, w) = ({expected}, {expected}) but got ({h}, {w})."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Calculate the embeddings for each patch and ensure they form a sequence with
        # shape (batch, sequence_length, patch_embedddings_size)
        patch_embeddings: torch.Tensor = self.patcher(image_batch)
        patch_embeddings = patch_embeddings.flatten(2, 3).transpose(1, 2)

        # Expand the class token so it can be concatenated to each embedding in the
        # batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        classed_patch_embeddings = torch.concat(
            [batch_class_token, patch_embeddings], dim=1
        )

        # Pass through the encoder layers
        encoder_output: torch.Tensor = self.encoder(classed_patch_embeddings)

        # Extract the output "embedding" associated with the input "[class]" token
        out_class_embedding = encoder_output[:, 0, :]

        # Pass the embedding through the head to perform classification
        return self.head(out_class_embedding)

    def fit(self, data_path: Path, epochs: int = 100, batch_size: int = 16):
        # Set the model to training mode such that Modules like Dropout and BatchNorm
        # behave appropriately during training
        self.train()


class Encoder(Module, MetaLogger):
    def __init__(
        self,
        sequence_length: int,
        patch_embedding_size: int,
        config: EncoderConfig,
    ):
        Module.__init__(self)
        MetaLogger.__init__(self)

        self.sequence_length = sequence_length
        self.patch_embedding_size = patch_embedding_size
        self.config = config

        # Learnable position embeddings. Each of these is added to the patch embedding
        # encode into the embedding the position of the patch.
        self.position_embeddings = Parameter(
            torch.empty(1, sequence_length, patch_embedding_size).normal_(std=0.02)
        )

        # Create the encoder layers
        layers: OrderedDict[str, Module] = OrderedDict()
        for layer_index in range(config.num_layers):
            layers[f"encoder_layer_{layer_index}"] = EncoderBlock(
                patch_embedding_size,
                config.dropout,
                config.attention_config,
                config.mlp_config,
            )
        self.layers = Sequential(layers)

        self.dropout = Dropout(config.dropout)
        self.norm_layer = LayerNorm(patch_embedding_size, eps=1e-6)

    def _check_input(self, x: torch.Tensor):
        # Check the input has the correct number of dimensions
        if x.dim() != 3:
            msg = f"Expected a 3 dimensional input but got {x.dim()}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Check the input is shaped correctly
        _, s, e = x.shape
        if s != self.sequence_length or e != self.patch_embedding_size:
            msg = (
                f"Expected shape of (n, s, e) = (_, {self.sequence_length}, "
                f"{self.patch_embedding_size}) but got (_, {s}, {e})."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        self._check_input(patch_embeddings)

        embeddings = patch_embeddings + self.position_embeddings

        # Pass the embeddings through the encoder layers
        embeddings = self.dropout(embeddings)
        output_embeddings = self.layers(embeddings)
        return self.norm_layer(output_embeddings)


class EncoderBlock(Module, MetaLogger):
    def __init__(
        self,
        patch_embedding_size: int,
        dropout: float,
        attention_config: AttentionConfig,
        mlp_config: MLPConfig,
    ):
        Module.__init__(self)
        MetaLogger.__init__(self)

        self.patch_embedding_size = patch_embedding_size
        self.attention_config = attention_config
        self.mlp_config = mlp_config

        # Define the attention block
        #   - The normalisation layer ensures that each embeddings is normalised input a
        #     normal distribution.
        #   - Multi-headed attention correlates relationships between input embeddings
        #   - Dropout provides randomisation during training to improve
        #     generalisability.
        self.norm_layer_1 = LayerNorm(patch_embedding_size, eps=1e-6)
        self.self_attention = MultiheadAttention(
            patch_embedding_size,
            attention_config.num_heads,
            dropout=attention_config.dropout,
            batch_first=True,
        )
        self.dropout = Dropout(dropout)

        # Multi-layer perceptron
        #   - Layer norm to ensure output embeddings fit the standard normal
        #     distribution.
        #   - Multi-layer perceptron is used to extract deeper features/relationships.
        self.norm_layer_2 = LayerNorm(patch_embedding_size, eps=1e-6)
        self.mlp = MLPBlock(patch_embedding_size, dropout, mlp_config)

    def _check_input(self, x: torch.Tensor):
        # Check the input has the correct number of dimensions
        if x.dim() != 3:
            msg = f"Expected a 3 dimensional input but got {x.dim()}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Check the embeddings size is correct
        if x.shape[2] == self.patch_embedding_size:
            msg = (
                f"Expected an embedding size of {self.patch_embedding_size} but got"
                f" {x.shape[2]}."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Pass the embeddings through a multi-headed attention mechanism
        # This allows the network to determine relationships between the various
        # embeddings in the sequence
        # It should be noted that the multi-head attention block applies the Q, K and V
        # projections despite requiring separate inputs for each
        norm_embed = self.norm_layer_1(embeddings)
        atten_embed, _ = self.self_attention.forward(
            norm_embed, norm_embed, norm_embed, need_weights=False
        )
        atten_embed = self.dropout(atten_embed)

        # Add a skip connection
        atten_with_skip_embed = atten_embed + embeddings

        # Apply the multi-layer perceptron
        mlp_input = self.norm_layer_2(atten_with_skip_embed)
        mlp_output = self.mlp(mlp_input)

        # Apply a skip connection
        return mlp_output + atten_with_skip_embed


class MLPBlock(Sequential):
    def __init__(self, input_size: int, dropout: float, config: MLPConfig):
        layers: list[Module] = []

        # Construct the layers of the MLP
        # This ensures that the output shape is equal to the input shape
        current_input_size = input_size
        for size in config.hidden_layer_sizes:
            layers.append(Linear(current_input_size, size, bias=True))
            layers.append(GELU())
            layers.append(Dropout(dropout))
            current_input_size = size
        layers.append(Linear(current_input_size, input_size))
        layers.append(Dropout(dropout))

        super().__init__(*layers)

        # Ensure the layers are initialised in the desired fashion
        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias, std=1e-6)


if __name__ == "__main__":
    import torchvision

    # Instantiate the model
    model = ViT(ViTConfig())

    # Load in a dataset
    cifar100_data = torchvision.datasets.CIFAR100(
        "/mnt/data/documents/code/ml-projects/ViT-TensorRT/datasets/cifar100"
    )
    data_loader = torch.utils.data.DataLoader(cifar100_data, batch_size=16)

    print(data_loader)
