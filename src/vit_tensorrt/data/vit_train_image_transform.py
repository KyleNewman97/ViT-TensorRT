import torch
from torchvision.transforms import v2


class ViTTrainImageTransform(v2.Compose):
    """
    The transform to apply to images during training. This includes various
    augmentations to help address biases in the training data.
    """

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
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
