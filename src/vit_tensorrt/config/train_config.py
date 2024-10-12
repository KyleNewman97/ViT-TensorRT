import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


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
    learning_rate: float = Field(default=1e-4)

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
