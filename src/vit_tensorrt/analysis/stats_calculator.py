import torch
from torch.nn import functional
from torchvision.transforms import v2

laplacian_kernel = torch.tensor(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ],
    dtype=torch.float32,
)


class StatsCalculator:
    """
    A utility class for calculating metrics image metrics.
    """

    @staticmethod
    def calculate_blurriness(image: torch.Tensor) -> float:
        """
        Convolves a Laplacian filter over the entire image and then calculates the mean
        value of the image. This is intended to be a rough estimation of how blurry the
        image is.

        Parameters
        ----------
        image:
            The image to calculate the mean blurriness of. Assumes this is arranged as
            (channels, height, width). This image can either be and RGB image or a
            grayscale image.
        """
        if image.shape[0] == 3:
            grayscale_image = v2.functional.rgb_to_grayscale(image)
        elif image.dim() == 2:
            grayscale_image = image[None, ...]
        elif image.shape[0] == 1:
            grayscale_image = image
        else:
            msg = f"Image has {image.shape[0]} channels, but expected 1 or 3."
            raise RuntimeError(msg)

        grayscale_image = grayscale_image[None, ...]
        grayscale_image = grayscale_image.type(torch.float32)
        kernel = laplacian_kernel[None, None, ...]
        return functional.conv2d(grayscale_image, kernel).mean().item()
