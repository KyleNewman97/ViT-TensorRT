from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from vit_tensorrt.utils import MetaLogger
from vit_tensorrt.data.vit_train_image_transform import ViTTrainImageTransform
from vit_tensorrt.data.vit_predict_image_transform import ViTPredictImageTransform


class ViTDataset(Dataset, MetaLogger):
    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        num_classes: int,
        transform: (
            ViTTrainImageTransform | ViTPredictImageTransform
        ) = ViTTrainImageTransform(),
    ):
        """
        Parameters
        ----------
        images_path:
            Path to a subset (train, valid, test) of the image data.

        labels_path:
            Path to a subset (train, valid, test) of the label data.

        num_classes:
            The number of classes classification will be run on.

        transform:
            The transform that should be applied to the image prior to inference.
        """
        MetaLogger.__init__(self)

        self.images_path = images_path
        self.labels_path = labels_path
        self.num_classes = num_classes
        self.transform = transform

        self.logger.info(f"Images path: {images_path}")
        self.logger.info(f"Labels path: {labels_path}")

        images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        labels = list(labels_path.glob("*.txt"))

        self.logger.info(f"Found {len(images)} images and {len(labels)} labels.")

        self.samples = self._get_file_pairs(images, labels)
        self.samples = self._filter_for_valid_labels(self.samples)

        self.logger.info(f"{len(self.samples)} image and label pairs exist.")

    def _get_file_pairs(
        self, images: list[Path], labels: list[Path]
    ) -> list[tuple[Path, Path]]:
        """
        Keeps only image and label files that form a pair. A pair requires the image
        and label file to have the same stem.

        Output
        ------
        samples:
            A list of samples. Each sample is structured as `(image_file, label_file)`.
        """
        stem_to_image_file = {i.stem: i for i in images}
        stem_to_label_file = {l.stem: l for l in labels}

        samples: list[tuple[Path, Path]] = []

        for stem, image_file in stem_to_image_file.items():
            if stem not in stem_to_label_file:
                self.logger.warning(f"Missing label file for {image_file}.")
                continue
            samples.append((image_file, stem_to_label_file[stem]))

        return samples

    def _filter_for_valid_labels(
        self, samples: list[tuple[Path, Path]]
    ) -> list[tuple[Path, Path]]:
        """
        Filter out samples with invalid labels.
        """
        expected_classes = set([f"{i}" for i in range(self.num_classes)])

        filtered_samples: list[tuple[Path, Path]] = []
        for sample in samples:
            with open(sample[1], "r") as fp:
                label = fp.read().strip()

                if label not in expected_classes:
                    self.logger.warning(f"Invalid label {label} in {sample[1]}.")
                    continue

                filtered_samples.append(sample)

        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image_file, label_file = self.samples[idx]

        # Load in the image and pre-process it
        pil_image = Image.open(image_file)
        image = pil_to_tensor(pil_image)
        image = self.transform(image)

        # Load in the label
        with open(label_file, "r") as fp:
            class_id = int(fp.read().strip())

        if 2 < self.num_classes:
            label = torch.zeros((self.num_classes,))
            label[class_id] = 1
        else:
            label = torch.zeros((1,))
            label[0] = class_id

        return image, label
