import os
from datetime import datetime
from pathlib import Path

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
    num_classes: int = Field(default=1000, ge=2)

    encoder_config: EncoderConfig = Field(default_factory=EncoderConfig)


class TrainConfig(BaseModel):
    """
    Defines the data to train on as well as the training configuration.
    """

    data_path: Path
    run_name: str = Field(
        default_factory=lambda: datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    )
    epochs: int = Field(default=100)
    batch_size: int = Field(default=16)
    learning_rate: float = Field(default=1e-3)

    @property
    def log_dir(self) -> Path:
        log_dir = Path(os.getcwd()) / "runs" / self.run_name
        log_dir.mkdir(exist_ok=True, parents=True)
        return log_dir

    @property
    def train_path(self) -> Path:
        return self.data_path / "train"

    @property
    def valid_path(self) -> Path:
        return self.data_path / "valid"

    @property
    def train_images_path(self) -> Path:
        return self.train_path / "images"

    @property
    def train_labels_path(self) -> Path:
        return self.train_path / "labels"

    @property
    def valid_images_path(self) -> Path:
        return self.valid_path / "images"

    @property
    def valid_labels_path(self) -> Path:
        return self.valid_path / "labels"
