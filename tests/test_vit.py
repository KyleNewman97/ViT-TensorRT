import torch
from unittest.mock import MagicMock, patch

from vit_tensorrt import ViT


class TestViT:
    @patch("torch.cuda.is_available", lambda: True)
    def test_init_gpu_available(self):
        """
        Test that the default device is set correctly when CUDA is available and we want
        to run on the first GPU.
        """
        # mock the call to set the device
        torch.set_default_device = MagicMock()

        device = "cuda:0"
        ViT(16, device)

        torch.set_default_device.assert_called_once_with(device)

    @patch("torch.cuda.is_available", lambda: False)
    def test_init_gpu_unavailable(self):
        """
        Test that the default device is set correctly when CUDA is unavailable and we
        want  to run on the first GPU.
        """
        # mock the call to set the device
        torch.set_default_device = MagicMock()

        ViT(16, "cuda:0")

        torch.set_default_device.assert_called_once_with("cpu")

    def test_init_cpu(self):
        """
        Test that the default device is set correctly when we want to run on the CPU.
        """
        # mock the call to set the device
        torch.set_default_device = MagicMock()

        device = "cpu"
        ViT(16, device)

        torch.set_default_device.assert_called_once_with(device)
