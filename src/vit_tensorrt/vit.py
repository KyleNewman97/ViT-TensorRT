from collections import OrderedDict
from pathlib import Path

import torch
import numpy as np
from torch.amp import autocast, GradScaler
from torch.nn import (
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    init,
    GELU,
    LayerNorm,
    Linear,
    Module,
    MultiheadAttention,
    Parameter,
    Sequential,
)
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vit_tensorrt.data import ViTDataset
from vit_tensorrt.utils import MetaLogger
from vit_tensorrt.config import (
    AttentionConfig,
    EncoderConfig,
    MLPConfig,
    TrainConfig,
    ViTConfig,
)


class ViT(Module, MetaLogger):
    def __init__(
        self,
        config: ViTConfig = ViTConfig(),
        device: str = "cuda:0",
    ):
        Module.__init__(self)
        MetaLogger.__init__(self)

        self.config = config
        self._set_compute_device(device)

        # Convolves over the image extracting patches of size `patch_size * patch_size`.
        # This ensures the output embedding contains `patch_embedding_size` channels.
        self.patcher = Conv2d(
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
        self.class_token = Parameter(torch.zeros(1, 1, config.patch_embedding_size))
        sequence_length += 1

        self.encoder = Encoder(
            sequence_length,
            config.patch_embedding_size,
            config.encoder_config,
        )

        # Initialise the "head" linear layer
        num_neurons = config.num_classes if config.num_classes > 2 else 1
        self.head = Linear(config.patch_embedding_size, num_neurons)

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
            BCEWithLogitsLoss() if self.config.num_classes == 2 else CrossEntropyLoss()
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


class Encoder(Module, MetaLogger):
    def __init__(
        self,
        sequence_length: int,
        patch_embedding_size: int,
        config: EncoderConfig,
    ):
        Module.__init__(self)
        MetaLogger.__init__(self)

        self.sequence_length = sequence_length
        self.patch_embedding_size = patch_embedding_size
        self.config = config

        # Learnable position embeddings. Each of these is added to the patch embedding
        # encode into the embedding the position of the patch.
        self.position_embeddings = Parameter(
            torch.empty(1, sequence_length, patch_embedding_size).normal_(std=0.02)
        )

        # Create the encoder layers
        layers: OrderedDict[str, Module] = OrderedDict()
        for layer_index in range(config.num_layers):
            layers[f"encoder_layer_{layer_index}"] = EncoderBlock(
                patch_embedding_size,
                config.dropout,
                config.attention_config,
                config.mlp_config,
            )
        self.layers = Sequential(layers)

        self.dropout = Dropout(config.dropout)
        self.norm_layer = LayerNorm(patch_embedding_size, eps=1e-6)

    def _check_input(self, x: torch.Tensor):
        # Check the input has the correct number of dimensions
        if x.dim() != 3:
            msg = f"Expected a 3 dimensional input but got {x.dim()}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Check the input is shaped correctly
        _, s, e = x.shape
        if s != self.sequence_length or e != self.patch_embedding_size:
            msg = (
                f"Expected shape of (n, s, e) = (_, {self.sequence_length}, "
                f"{self.patch_embedding_size}) but got (_, {s}, {e})."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        self._check_input(patch_embeddings)

        embeddings = patch_embeddings + self.position_embeddings

        # Pass the embeddings through the encoder layers
        embeddings = self.dropout(embeddings)
        output_embeddings = self.layers(embeddings)
        return self.norm_layer(output_embeddings)


class EncoderBlock(Module, MetaLogger):
    def __init__(
        self,
        patch_embedding_size: int,
        dropout: float,
        attention_config: AttentionConfig,
        mlp_config: MLPConfig,
    ):
        Module.__init__(self)
        MetaLogger.__init__(self)

        self.patch_embedding_size = patch_embedding_size
        self.attention_config = attention_config
        self.mlp_config = mlp_config

        # Define the attention block
        #   - The normalisation layer ensures that each embeddings is normalised input a
        #     normal distribution.
        #   - Multi-headed attention correlates relationships between input embeddings
        #   - Dropout provides randomisation during training to improve
        #     generalisability.
        self.norm_layer_1 = LayerNorm(patch_embedding_size, eps=1e-6)
        self.self_attention = MultiheadAttention(
            patch_embedding_size,
            attention_config.num_heads,
            dropout=attention_config.dropout,
            batch_first=True,
        )
        self.dropout = Dropout(dropout)

        # Multi-layer perceptron
        #   - Layer norm to ensure output embeddings fit the standard normal
        #     distribution.
        #   - Multi-layer perceptron is used to extract deeper features/relationships.
        self.norm_layer_2 = LayerNorm(patch_embedding_size, eps=1e-6)
        self.mlp = MLPBlock(patch_embedding_size, dropout, mlp_config)

    def _check_input(self, x: torch.Tensor):
        # Check the input has the correct number of dimensions
        if x.dim() != 3:
            msg = f"Expected a 3 dimensional input but got {x.dim()}."
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Check the embeddings size is correct
        if x.shape[2] == self.patch_embedding_size:
            msg = (
                f"Expected an embedding size of {self.patch_embedding_size} but got"
                f" {x.shape[2]}."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Pass the embeddings through a multi-headed attention mechanism
        # This allows the network to determine relationships between the various
        # embeddings in the sequence
        # It should be noted that the multi-head attention block applies the Q, K and V
        # projections despite requiring separate inputs for each
        norm_embed = self.norm_layer_1(embeddings)
        atten_embed, _ = self.self_attention.forward(
            norm_embed, norm_embed, norm_embed, need_weights=False
        )
        atten_embed = self.dropout(atten_embed)

        # Add a skip connection
        atten_with_skip_embed = atten_embed + embeddings

        # Apply the multi-layer perceptron
        mlp_input = self.norm_layer_2(atten_with_skip_embed)
        mlp_output = self.mlp(mlp_input)

        # Apply a skip connection
        return mlp_output + atten_with_skip_embed


class MLPBlock(Sequential):
    def __init__(self, input_size: int, dropout: float, config: MLPConfig):
        layers: list[Module] = []

        # Construct the layers of the MLP
        # This ensures that the output shape is equal to the input shape
        current_input_size = input_size
        for size in config.hidden_layer_sizes:
            layers.append(Linear(current_input_size, size, bias=True))
            layers.append(GELU())
            layers.append(Dropout(dropout))
            current_input_size = size
        layers.append(Linear(current_input_size, input_size))
        layers.append(Dropout(dropout))

        super().__init__(*layers)

        # Ensure the layers are initialised in the desired fashion
        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias, std=1e-6)


if __name__ == "__main__":
    from pathlib import Path

    # Instantiate the model
    encoder_config = EncoderConfig(num_layers=8, dropout=0.1)
    config = ViTConfig(image_size=320, num_classes=2, encoder_config=encoder_config)
    model = ViT(config, device="cuda:0")

    train_config = TrainConfig(
        data_path=Path("datasets/cat-dog"),
        batch_size=64,
        learning_rate=1e-4,
    )
    model.fit(train_config)
