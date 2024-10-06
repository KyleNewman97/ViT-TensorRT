from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vit_tensorrt.config import TrainConfig, ViTConfig
from vit_tensorrt.data import ViTDataset, ViTPredictImageTransform
from vit_tensorrt.model.encoder import Encoder
from vit_tensorrt.utils import MetaLogger


class ViT(nn.Module, MetaLogger):
    def __init__(
        self,
        config: ViTConfig = ViTConfig(),
        device: str = "cuda:0",
    ):
        nn.Module.__init__(self)
        MetaLogger.__init__(self)

        self.config = config
        self._set_compute_device(device)

        # Convolves over the image extracting patches of size `patch_size * patch_size`.
        # This ensures the output embedding contains `patch_embedding_size` channels.
        self.patcher = nn.Conv2d(
            3, config.patch_embedding_size, config.patch_size, config.patch_size
        )

        # Determine the number of embeddings after the conv layer
        sequence_length = (config.image_size // config.patch_size) ** 2

        # Create a learnable [class] token
        # We use the class token because encoders are squence-to-squence models, so the
        # input sequence length equals the output sequence length.
        #   - If we applied the linear layer to all sequence outputs it would fix the
        #     input image size.
        #   - If we applied it to a single element of the sequence, our output my be
        #     unfairly biased towards the corresponding patch.
        # By using the class token, we can use the output corresponding to it as the
        # input to the linear layer.
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.patch_embedding_size))
        sequence_length += 1

        self.encoder = Encoder(
            sequence_length,
            config.patch_embedding_size,
            config.encoder_config,
        )

        # Initialise the "head" linear layer
        num_neurons = config.num_classes if config.num_classes > 2 else 1
        self.head = nn.Linear(config.patch_embedding_size, num_neurons)

    def _set_compute_device(self, device: str):
        if "cuda" in device and torch.cuda.is_available():
            self.device = device
        elif "cuda" in device and not torch.cuda.is_available():
            self.logger.warning("CUDA device not found - running on CPU.")
            self.device = "cpu"
        else:
            self.device = device
        torch.set_default_device(self.device)

        self.logger.info(f"Set default device to {torch.get_default_device()}")

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass of the model.

        Parameters
        ----------
        x: `torch.Tensor`
            An input batch of images shaped as `(batch_size, channel, height, width)`.
        """

        # Ensure the input image has the right shape
        n, _, h, w = image_batch.shape
        if h != self.config.image_size or w != self.config.image_size:
            expected = self.config.image_size
            msg = f"Expected (h, w) = ({expected}, {expected}) but got ({h}, {w})."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Calculate the embeddings for each patch and ensure they form a sequence with
        # shape (batch, sequence_length, patch_embedddings_size)
        patch_embeddings: torch.Tensor = self.patcher(image_batch)
        patch_embeddings = patch_embeddings.flatten(2, 3).transpose(1, 2)

        # Expand the class token so it can be concatenated to each embedding in the
        # batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        classed_patch_embeddings = torch.concat(
            [batch_class_token, patch_embeddings], dim=1
        )

        # Pass through the encoder layers
        encoder_output: torch.Tensor = self.encoder(classed_patch_embeddings)

        # Extract the output "embedding" associated with the input "[class]" token
        out_class_embedding = encoder_output[:, 0, :]

        # Pass the embedding through the head to perform classification
        return self.head(out_class_embedding)

    def _prepare_data(self, config: TrainConfig) -> tuple[DataLoader, DataLoader]:
        # Create the training data loader
        train_dataset = ViTDataset(
            config.train_images_path,
            config.train_labels_path,
            self.config.num_classes,
            self.config.image_size,
            self.config.image_size,
        )
        train_loader = DataLoader(
            train_dataset,
            config.batch_size,
            shuffle=True,
            generator=torch.Generator(device=self.device),
        )
        self.logger.info(f"Training with {len(train_dataset)} samples.")

        # Create the validation data loader
        valid_dataset = ViTDataset(
            config.valid_images_path,
            config.valid_labels_path,
            self.config.num_classes,
            self.config.image_size,
            self.config.image_size,
        )
        valid_loader = DataLoader(
            valid_dataset,
            config.batch_size,
            shuffle=True,
            generator=torch.Generator(device=self.device),
        )
        self.logger.info(f"Validating with {len(valid_dataset)} samples.")

        return train_loader, valid_loader

    def fit(self, config: TrainConfig):
        # Create datasets and loaders for train and valid
        train_loader, valid_loader = self._prepare_data(config)

        # Initialise loss, optimiser and learning rate scheduler
        loss_func = (
            nn.BCEWithLogitsLoss()
            if self.config.num_classes == 2
            else nn.CrossEntropyLoss()
        )
        optimiser = Adam(self.parameters(), lr=config.learning_rate)
        scheduler = ChainedScheduler(
            [
                LinearLR(optimiser, total_iters=3),
                CosineAnnealingLR(optimiser, T_max=config.epochs),
            ],
            optimiser,
        )

        # Used for scaling gradients in fp16 layers where they may underflow precision
        # limits
        scaler = GradScaler(device=self.device)

        writer = SummaryWriter(config.log_dir)

        loss = 0
        best_v_loss = np.inf
        running_loss_batches = 4
        self.logger.info(f"Training with config: {config}")
        for epoch in range(config.epochs):
            self.logger.info(f"Epoch: {epoch}")

            # Run the training epoch
            self.train()
            running_loss = 0
            tqdm_iterator = tqdm(train_loader, ncols=88)
            tqdm_iterator.set_description_str("Train")
            for idx, (images, labels) in enumerate(tqdm_iterator):
                # Zero the gradients - this is required on each mini-batch
                optimiser.zero_grad()

                # Use automatic mixed precision to reduce memory footprint
                with autocast(device_type=self.device):
                    # Make predictions for this batch and calculate the loss
                    outputs: torch.Tensor = self(images)
                    loss: torch.Tensor = loss_func(outputs, labels)

                # Backprop
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

                tqdm_iterator.set_postfix_str(f"loss={loss.item():.4}")

                # Update the running loss only on the last `running_loss_batches` mini-
                # batches
                if len(train_loader) - running_loss_batches <= idx:
                    running_loss += loss

            # Update the learning rate scheduler
            scheduler.step()

            tqdm_iterator.close()

            # Run over the validation dataset
            self.eval()
            running_vloss = 0
            num_correct = 0
            with torch.no_grad():
                tqdm_iterator = tqdm(valid_loader, ncols=88)
                tqdm_iterator.set_description_str("Valid")
                for images, labels in tqdm_iterator:
                    outputs: torch.Tensor = self(images)
                    loss = loss_func(outputs, labels)
                    running_vloss += loss

                    # Calculate the number of correct classifications
                    if 2 < self.config.num_classes:
                        classifications = torch.nn.functional.softmax(outputs, dim=1)
                        pred_classes = classifications.argmax(dim=1)
                        true_classes: torch.Tensor = labels.argmax(dim=1)
                        num_correct += (pred_classes == true_classes).sum()
                    else:
                        classifications = torch.nn.functional.sigmoid(outputs)
                        pred_classes = 0.5 < classifications
                        true_classes: torch.Tensor = labels == 1
                        num_correct += (pred_classes == true_classes).sum()
                tqdm_iterator.close()

            # Calculate metrcs
            t_loss = (running_loss / running_loss_batches).item()
            v_loss = (running_vloss / len(valid_loader)).item()
            self.logger.info(f"Train loss: {t_loss:.4} Valid loss: {v_loss:.4}")
            v_accuracy = num_correct / len(valid_loader.dataset)
            self.logger.info(f"Valid accuracy: {v_accuracy.item()}")

            # Log metrics to the summary writer
            writer.add_scalar("Loss/train", t_loss, epoch)
            writer.add_scalar("Loss/valid", v_loss, epoch)
            writer.add_scalar("Acc/valid", v_accuracy, epoch)
            writer.add_scalar("Optim/lr", scheduler.get_last_lr()[0], epoch)
            writer.flush()

            # Save the model
            if v_loss < best_v_loss:
                best_v_loss = v_loss
                self.logger.info("Saving new best model.")
                self.save(config.log_dir / "best.pt")
            self.save(config.log_dir / "last.pt")

        # Clean up at the end of training
        writer.close()

    def _prepare_eval_data(self, data_path: Path) -> ViTDataset:
        im_size = self.config.image_size
        transform = ViTPredictImageTransform(im_size, im_size)
        return ViTDataset(
            data_path / "images",
            data_path / "labels",
            self.config.num_classes,
            transform,
        )

    def evaluate(
        self, data_path: Path
    ) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
        dataset = self._prepare_eval_data(data_path)

        image: torch.Tensor = None
        label: torch.Tensor = None

        correct, incorrect = [], []

        # Inference loop
        self.eval()
        with torch.no_grad():
            iterator = tqdm(dataset, ncols=88, desc="Evaluate")
            for idx, (image, label) in enumerate(iterator):
                # Run the image through the model
                prediction: torch.Tensor = self(image.unsqueeze(0))

                # Determine the predicted class
                if 2 < self.config.num_classes:
                    classifications = torch.nn.functional.softmax(prediction, dim=1)
                    pred_class = classifications.argmax(dim=1)
                    true_class: torch.Tensor = label.argmax(dim=1)
                else:
                    classifications = torch.nn.functional.sigmoid(prediction)
                    pred_class = 0.5 < classifications
                    true_class: torch.Tensor = label == 1

                if pred_class != true_class:
                    incorrect.append(dataset.samples[idx])
                else:
                    correct.append(dataset.samples[idx])

        return correct, incorrect

    def save(self, file: Path):
        """
        Saves the model to the specified location. By convention this location should
        end with ".pt".
        """

        torch.save(
            {
                "model": self.state_dict(),
                "config": self.config.model_dump(),
                "device": self.device,
            },
            file,
        )

    @classmethod
    def load(cls, file: Path) -> "ViT":
        """
        Load the model from the specified location.
        """
        # Load the previous state
        model_state = torch.load(file, weights_only=True)
        config = ViTConfig(**model_state["config"])
        device = model_state["device"]

        # Initialise the model
        model = cls(config, device)
        model.load_state_dict(model_state["model"])
        model.eval()

        return model
