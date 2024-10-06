from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
import torch
import numpy as np
from PIL import Image
from torch import Tensor

from vit_tensorrt.data import ViTDataset, ViTTrainImageTransform


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
        height, width = 256, 512
        transform = ViTTrainImageTransform(height, width)
        dataset = ViTDataset(self.images_path, self.labels_path, num_classes, transform)
        image, label = dataset[0]

        assert isinstance(image, Tensor)
        assert image.shape == (3, height, width)
        assert isinstance(label, Tensor)
        assert label.shape == (1,)
