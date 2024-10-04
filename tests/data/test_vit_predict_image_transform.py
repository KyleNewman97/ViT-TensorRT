import torch

from vit_tensorrt.data import ViTPredictImageTransform


class TestViTPredictImageTransform:
    def test_init(self):
        transform = ViTPredictImageTransform(height=256, width=512)
        assert isinstance(transform, ViTPredictImageTransform)

    def test_call(self):
        """
        Test that the transform applies the correct operations to the image.
        """
        image = torch.randint(0, 255, (3, 100, 120))

        transform = ViTPredictImageTransform(height=256, width=512)
        out_image: torch.Tensor = transform(image)

        assert isinstance(out_image, torch.Tensor)
        assert out_image.shape == (3, transform.height, transform.width)
        assert out_image.max() < 100
