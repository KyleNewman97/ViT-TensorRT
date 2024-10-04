from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor

from vit_tensorrt.utils import MetaLogger


class OptionalDivision:
    """
    Divides the image's content by `255` if the max value in the image is greater than
    1.
    """

    def __call__(self, image: Tensor) -> Tensor:
        if 1 < image.max():
            return image / 255
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ViTImageTransform(v2.Compose):
    def __init__(self, height: int = 768, width: int = 768):
        v2.Compose.__init__(
            self,
            [
                v2.Resize((height, width)),
                v2.ToDtype(torch.float32, True),
                OptionalDivision(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )


class ViTTrainImageTransform(v2.Compose):
    def __init__(self, height: int = 768, width: int = 768):

        # v2.RandomChoice(
        #     [
        #         v2.GaussianNoise(mean=0, sigma=5e-2),
        #         v2.RandomInvert(p=0.5),
        #         v2.RandomGrayscale(p=0.5),
        #         v2.RandomCrop((height // 10, width // 10)),
        #     ]
        # ),
        v2.Compose.__init__(
            self,
            [
                v2.Resize((height, width)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(90),
                v2.ToDtype(torch.float32, True),
                OptionalDivision(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )


class ViTDataset(Dataset, MetaLogger):
    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        num_classes: int,
        image_height: int = 768,
        image_width: int = 768,
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

        image_height:
            The height of the image to pass to the network, in pixels.

        image_width:
            The width of the image to pass to the network, in pixels.
        """
        MetaLogger.__init__(self)
        self.transform = ViTTrainImageTransform(image_height, image_width)

        self.images_path = images_path
        self.labels_path = labels_path
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width

        self.logger.info(f"Images path: {images_path}")
        self.logger.info(f"Labels path: {labels_path}")

        images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        labels = list(labels_path.glob("*.txt"))

        self.logger.info(f"Found {len(images)} images and {len(labels)} labels.")

        self.samples = self._get_file_pairs(images, labels)
        self.samples = self._filter_for_valid_labels(self.samples)

        self.logger.info(f"{len(self.samples)} valid image and label pairs exist.")

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
