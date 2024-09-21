from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Resize


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
        resizer = Resize((new_height, new_width))
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


class ViTDataset(Dataset):
    def __init__(self, data_path: Path, labels, transform=None):
        self.data_path = data_path
        self.image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
        self.labels = labels
        self.transform = transform

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     sample = self.data[idx]
    #     label = self.labels[idx]

    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample, label
