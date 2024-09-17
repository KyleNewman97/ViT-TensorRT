from pydantic import BaseModel, Field


class AttentionConfig(BaseModel):
    num_heads: int = Field(default=16)
    dropout: float = Field(default=0)


class MLPConfig(BaseModel):
    hidden_layer_sizes: list[int] = Field(default=[3072])


class EncoderConfig(BaseModel):
    num_layers: int = Field(default=24)
    dropout: float = Field(default=0)

    attention_config: AttentionConfig = Field(default_factory=AttentionConfig)
    mlp_config: MLPConfig = Field(default_factory=MLPConfig)


class ViTConfig(BaseModel):
    image_size: int = Field(default=768)
    patch_size: int = Field(default=16)
    patch_embedding_size: int = Field(default=768)
    num_classes: int = Field(default=1000)

    encoder_config: EncoderConfig = Field(default_factory=EncoderConfig)
