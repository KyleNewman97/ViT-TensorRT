from pathlib import Path

from vit_tensorrt.config import EncoderConfig, TrainConfig, ViTConfig
from vit_tensorrt.model import ViT


if __name__ == "__main__":
    # Instantiate the model
    encoder_config = EncoderConfig(num_layers=8, dropout=0.1)
    config = ViTConfig(image_size=320, num_classes=2, encoder_config=encoder_config)
    model = ViT(config)

    # Train the model
    train_config = TrainConfig(
        data_path=Path("datasets/cat-dog"),
        batch_size=64,
        learning_rate=1e-4,
    )
    model.fit(train_config)

    # Evaluate the model
    model = ViT.load("runs/10-03-2024_13-22-49/last.pt")
    model.evaluate(Path("datasets/cat-dog/valid"))
