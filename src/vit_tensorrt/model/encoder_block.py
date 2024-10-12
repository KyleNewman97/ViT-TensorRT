import torch
from torch import nn

from vit_tensorrt.config import AttentionConfig, MLPConfig
from vit_tensorrt.model.mlp_block import MLPBlock
from vit_tensorrt.utils import MetaLogger


class EncoderBlock(nn.Module, MetaLogger):
    def __init__(
        self,
        patch_embedding_size: int,
        dropout: float,
        attention_config: AttentionConfig,
        mlp_config: MLPConfig,
    ):
        nn.Module.__init__(self)
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
        self.norm_layer_1 = nn.LayerNorm(patch_embedding_size, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            patch_embedding_size,
            attention_config.num_heads,
            dropout=attention_config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Multi-layer perceptron
        #   - Layer norm to ensure output embeddings fit the standard normal
        #     distribution.
        #   - Multi-layer perceptron is used to extract deeper features/relationships.
        self.norm_layer_2 = nn.LayerNorm(patch_embedding_size, eps=1e-6)
        self.mlp = MLPBlock(patch_embedding_size, dropout, mlp_config)

    def _check_input(self, x: torch.Tensor):
        if torch.onnx.is_in_onnx_export():
            return

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
        self._check_input(embeddings)

        # Pass the embeddings through a multi-headed attention mechanism
        # This allows the network to determine relationships between the various
        # embeddings in the sequence
        # It should be noted that the multi-head attention block applies the Q, K and V
        # projections despite requiring separate inputs for each
        norm_embed: torch.Tensor = self.norm_layer_1(embeddings)
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
