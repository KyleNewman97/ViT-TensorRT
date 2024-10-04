import torch

from vit_tensorrt.data import ViTTrainImageTransform


class TestViTTrainImageTransform:
    def test_init(self):
        transform = ViTTrainImageTransform(768, 768)
        assert isinstance(transform, ViTTrainImageTransform)

    def test_transform(self):
        """
        Tests that the transforms are applied correctly.
        """
        height, width = 768, 768
        transform = ViTTrainImageTransform(height, width)

        # Try to apply the transforms to an input image
        image = torch.randint(0, 255, (3, 320, 320), dtype=torch.uint8)
        result = transform(image)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, height, width)
        assert result.dtype == torch.float32
        assert result.max() < 100
