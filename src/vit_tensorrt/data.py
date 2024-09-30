from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor

from vit_tensorrt.utils import MetaLogger


class Letterbox:
    """
    A PyTorch transform to letterbox an image to the specified `width` and `height`.
    Specifically, this scales the image to fit into the desired output size whilst
    maintaining the image's original aspect ratio. The scaled image is then centered
    leaving gray padding in the other regions.
    """

    def __init__(self, height: int = 768, width: int = 768):
        self.out_height = height
        self.out_width = width

    def __call__(self, image: Tensor) -> Tensor:
        """
        Parameters
        ----------
        image:
            The image to letterbox. The input must be shaped as [C, H, W].

        Returns
        -------
        letterboxed_image:
            The letterboxed image.
        """
        # Determine how much the image should be scaled up by
        _, height, width = image.shape
        height_ratio = self.out_height / height
        width_ratio = self.out_width / width
        scale = width_ratio if width_ratio < height_ratio else height_ratio

        # Resize the image to the correct scale
        new_height = int(scale * height)
        new_width = int(scale * width)
        resizer = v2.Resize((new_height, new_width))
        image = resizer(image)

        # Overlay on the output image size leaving padding
        top = (self.out_height - new_height) // 2
        bottom = top + new_height
        left = (self.out_width - new_width) // 2
        right = left + new_width
        out_image = torch.ones((3, self.out_height, self.out_width)) * 127
        out_image[:, top:bottom, left:right] = image

        return out_image

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(height={self.out_height},"
            f" width={self.out_width})"
        )


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
                Letterbox(height, width),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, True),
                OptionalDivision(),
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        self.transform = ViTImageTransform(image_height, image_width)

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
