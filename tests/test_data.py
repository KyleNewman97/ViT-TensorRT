from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
import torch
import numpy as np
from PIL import Image
from torch import Tensor

from vit_tensorrt.data import Letterbox, OptionalDivision, ViTDataset, ViTImageTransform


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


class TestOptionalDivision:
    @pytest.fixture
    def division(self) -> OptionalDivision:
        return OptionalDivision()

    def test_init(self, division: OptionalDivision):
        assert isinstance(division, OptionalDivision)

    def test_divide(self, division: OptionalDivision):
        """
        Test that we divide the image when the max value is above 1.
        """
        image = torch.randint(0, 255, (3, 320, 320), dtype=torch.uint8)
        result = division(image)

        assert isinstance(result, Tensor)
        assert 0 <= result.min()
        assert result.max() <= 1

    def test_skip_divide(self, division: OptionalDivision):
        """
        Test that we skip dividing the image when the max value is 1 or lower.
        """
        image = torch.ones((3, 320, 320), dtype=torch.uint8)
        result = division(image)

        assert isinstance(result, Tensor)
        assert (result == 1).all()


class TestViTImageTransform:
    def test_init(self):
        transform = ViTImageTransform(768, 768)
        assert isinstance(transform, ViTImageTransform)

    def test_transform(self):
        """
        Tests that the transforms are applied correctly.
        """
        height, width = 768, 768
        transform = ViTImageTransform(height, width)

        # Try to apply the transforms to an input image
        image = torch.randint(0, 255, (3, 320, 320), dtype=torch.uint8)
        result = transform(image)

        assert isinstance(result, Tensor)
        assert result.shape == (3, height, width)
        assert result.dtype == torch.float32
        assert result.max() < 100


class TestViTDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.images_path = self.temp_path / "images"
        self.labels_path = self.temp_path / "labels"

        self.images_path.mkdir(exist_ok=True, parents=True)
        self.labels_path.mkdir(exist_ok=True, parents=True)

        yield

        self.temp_dir.cleanup()

    def test_init_valid_dataset(self):
        """
        Test that we keep all valid samples when initialising the dataset.
        """
        # Make a single image file and label file that are valid
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("1")

        # Try to initialise the dataset
        dataset = ViTDataset(self.images_path, self.labels_path, 2)
        assert isinstance(dataset, ViTDataset)
        assert len(dataset) == 1

    def test_init_missing_label(self):
        """
        Test that we remove samples when we are missing the label.
        """
        # Make a single image file
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        # Try to initialise the dataset
        dataset = ViTDataset(self.images_path, self.labels_path, 2)
        assert isinstance(dataset, ViTDataset)
        assert len(dataset) == 0

    def test_init_invalid_label_class(self):
        """
        Test that we remove samples with invalid label file contents.
        """
        # Make a single image file
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("2")

        # Try to initialise the dataset
        dataset = ViTDataset(self.images_path, self.labels_path, 2)
        assert isinstance(dataset, ViTDataset)
        assert len(dataset) == 0

    def test_test_get_item(self):
        # Make a single image file and label file that are valid
        uuid = f"{uuid4()}"

        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")

        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("1")

        # Try to get a sample from the dataset
        num_classes = 2
        dataset = ViTDataset(self.images_path, self.labels_path, num_classes, 768, 768)
        image, label = dataset[0]

        assert isinstance(image, Tensor)
        assert image.shape == (3, dataset.image_height, dataset.image_width)
        assert isinstance(label, Tensor)
        assert label.shape == (num_classes,)
