from pydantic import BaseModel, Field

from vit_tensorrt.config.attention_config import AttentionConfig
from vit_tensorrt.config.mlp_config import MLPConfig


class EncoderConfig(BaseModel):
    num_layers: int = Field(default=24)
    dropout: float = Field(default=0)

    attention_config: AttentionConfig = Field(default_factory=AttentionConfig)
    mlp_config: MLPConfig = Field(default_factory=MLPConfig)
