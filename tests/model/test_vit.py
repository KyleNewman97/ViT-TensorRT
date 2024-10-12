import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from vit_tensorrt import ViT
from vit_tensorrt.config import ViTConfig


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
        ViT(ViTConfig(), device)

        torch.set_default_device.assert_called_once_with(device)

    @patch("torch.cuda.is_available", lambda: False)
    def test_init_gpu_unavailable(self):
        """
        Test that the default device is set correctly when CUDA is unavailable and we
        want  to run on the first GPU.
        """
        # mock the call to set the device
        torch.set_default_device = MagicMock()

        ViT(ViTConfig(), "cuda:0")

        torch.set_default_device.assert_called_once_with("cpu")

    def test_init_cpu(self):
        """
        Test that the default device is set correctly when we want to run on the CPU.
        """
        # mock the call to set the device
        torch.set_default_device = MagicMock()

        device = "cpu"
        ViT(ViTConfig(), device)

        torch.set_default_device.assert_called_once_with(device)

    def test_save_and_load(self):
        """
        Test that we can save and load the model.
        """

        model = ViT(ViTConfig(), "cpu")

        with TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "model.pt"
            model.save(model_file)

            loaded_model = ViT.load(model_file)

        assert isinstance(loaded_model, ViT)
        assert loaded_model.config == model.config
        assert loaded_model.device == model.device

    def test_to_onnx(self):
        """
        Test that we can convert to an ONNX file.
        """
        model = ViT(ViTConfig(), "cuda:0")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model.to_onnx(temp_path / "model.onnx")
