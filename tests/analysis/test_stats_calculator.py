import pytest
import torch
from vit_tensorrt.analysis import StatsCalculator


class TestStatsCalculator:
    @pytest.fixture
    def grayscale_image(self) -> torch.Tensor:
        return torch.tensor(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]]], dtype=torch.uint8
        )

    def test_calculate_blurriness_gray(self, grayscale_image: torch.Tensor):
        """
        Test that we can calculate the blurriness on a grayscale image.
        """
        blurriness = StatsCalculator.calculate_blurriness(grayscale_image)
        assert isinstance(blurriness, float)
        assert blurriness == 24
