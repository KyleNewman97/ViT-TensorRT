from pathlib import Path

import fiftyone as fo

from vit_tensorrt.config import EncoderConfig, TrainConfig, ViTConfig
from vit_tensorrt.model import ViT


if __name__ == "__main__":
    # # Instantiate the model
    # encoder_config = EncoderConfig(num_layers=8, dropout=0.1)
    # config = ViTConfig(image_size=320, num_classes=2, encoder_config=encoder_config)
    # model = ViT(config, device="cuda:0")

    # # Train the model
    # train_config = TrainConfig(
    #     data_path=Path("datasets/cat-dog"),
    #     batch_size=64,
    #     learning_rate=1e-4,
    # )
    # model.fit(train_config)

    # Evaluate the model
    model = ViT.load("runs/10-03-2024_13-22-49/last.pt")
    correct, incorrect = model.evaluate(Path("datasets/cat-dog/valid"))

    # Create the dataset
    dataset = fo.Dataset(name="Cat-Dog")

    for image_file, label_file in correct:
        sample = fo.Sample(filepath=image_file)

        with open(label_file, "r") as fp:
            class_id = int(fp.read().strip())
        class_name = "cat" if class_id == 0 else "dog"
        sample["ground_truth"] = fo.Classification(label=class_name)
        sample["predicted"] = fo.Classification(label=class_name)

        dataset.add_sample(sample)

    for image_file, label_file in incorrect:
        sample = fo.Sample(filepath=image_file)

        with open(label_file, "r") as fp:
            class_id = int(fp.read().strip())
        class_name = "cat" if class_id == 0 else "dog"
        predicted_class = "dog" if class_id == 0 else "cat"
        sample["ground_truth"] = fo.Classification(label=class_name)
        sample["predicted"] = fo.Classification(label=predicted_class)

        dataset.add_sample(sample)

    # Launch the FiftyOne app
    session = fo.launch_app(dataset)
    session.wait(wait=-1)
