from collections import OrderedDict

import torch
from torch import nn

from vit_tensorrt.config import EncoderConfig
from vit_tensorrt.model.encoder_block import EncoderBlock
from vit_tensorrt.utils import MetaLogger


class Encoder(nn.Module, MetaLogger):
    def __init__(
        self,
        sequence_length: int,
        patch_embedding_size: int,
        config: EncoderConfig,
    ):
        nn.Module.__init__(self)
        MetaLogger.__init__(self)

        self.sequence_length = sequence_length
        self.patch_embedding_size = patch_embedding_size
        self.config = config

        # Learnable position embeddings. Each of these is added to the patch embedding
        # encode into the embedding the position of the patch.
        self.position_embeddings = nn.Parameter(
            torch.empty(1, sequence_length, patch_embedding_size).normal_(std=0.02)
        )

        # Create the encoder layers
        layers: OrderedDict[str, EncoderBlock] = OrderedDict()
        for layer_index in range(config.num_layers):
            layers[f"encoder_layer_{layer_index}"] = EncoderBlock(
                patch_embedding_size,
                config.dropout,
                config.attention_config,
                config.mlp_config,
            )
        self.layers = nn.Sequential(layers)

        self.dropout = nn.Dropout(config.dropout)
        self.norm_layer = nn.LayerNorm(patch_embedding_size, eps=1e-6)

    def _check_input(self, x: torch.Tensor):
        if torch.onnx.is_in_onnx_export():
            return

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
