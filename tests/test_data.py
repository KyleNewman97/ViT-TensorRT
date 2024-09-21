import pytest
import torch
import numpy as np

from vit_tensorrt.data import Letterbox


class TestLetterbox:
    @pytest.fixture
    def letterbox(self) -> Letterbox:
        return Letterbox(768, 768)

    def test_init(self, letterbox: Letterbox):
        assert isinstance(letterbox, Letterbox)

    def test_letterbox_height_limit(self, letterbox: Letterbox):
        """
        Test that we can letterbox the image when the height is the limit.
        """

        image = torch.randint(0, 255, (3, 640, 320))
        out_image: torch.Tensor = letterbox(image)
        assert out_image.shape == (3, letterbox.out_height, letterbox.out_width)

    def test_letterbox_width_limit(self, letterbox: Letterbox):
        """
        Test that we can letterbox the image when the width is the limit.
        """

        image = torch.randint(0, 255, (3, 1080, 1920))
        out_image: torch.Tensor = letterbox(image)
        assert out_image.shape == (3, letterbox.out_height, letterbox.out_width)

    def test_letterbox_size_equal(self, letterbox: Letterbox):
        """
        Test that we can letterbox the image when no padding should be added
        """

        image = torch.randint(0, 255, (3, 320, 320))
        out_image: torch.Tensor = letterbox(image)
        assert out_image.shape == (3, letterbox.out_height, letterbox.out_width)
