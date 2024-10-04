from pydantic import BaseModel, Field

from vit_tensorrt.config.encoder_config import EncoderConfig


class ViTConfig(BaseModel):
    image_size: int = Field(default=768)
    patch_size: int = Field(default=16)
    patch_embedding_size: int = Field(default=768)
    num_classes: int = Field(default=1000, ge=2)

    encoder_config: EncoderConfig = Field(default_factory=EncoderConfig)
