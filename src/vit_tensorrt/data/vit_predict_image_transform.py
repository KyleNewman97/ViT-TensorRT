import torch
from torchvision.transforms import v2


class ViTPredictImageTransform(v2.Compose):
    """
    The transform to apply to images at prediction/evaluation time.
    """

    def __init__(self, height: int = 768, width: int = 768):
        self.height = height
        self.width = width

        v2.Compose.__init__(
            self,
            [
                v2.Resize((height, width)),
                v2.ToDtype(torch.float32, True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
