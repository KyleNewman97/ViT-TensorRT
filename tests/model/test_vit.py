from pathlib import Path
from tempfile import TemporaryDirectory

from vit_tensorrt import ViT
from vit_tensorrt.config import ViTConfig, EncoderConfig


class TestViT:
    def test_init(self):
        """
        Test that we can initialise the model correctly.
        """

        config = ViTConfig()
        model = ViT(config)

        assert isinstance(model, ViT)

    def test_save_and_load(self):
        """
        Test that we can save and load the model.
        """
        encoder_config = EncoderConfig(num_layers=4)
        config = ViTConfig(image_size=256, encoder_config=encoder_config)
        model = ViT(config)

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
        encoder_config = EncoderConfig(num_layers=4)
        config = ViTConfig(image_size=256, encoder_config=encoder_config)
        model = ViT(config)

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file = temp_path / "model.onnx"
            model.export_onnx(file)

            assert file.is_file()
