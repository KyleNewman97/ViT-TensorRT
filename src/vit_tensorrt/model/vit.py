from pathlib import Path

import torch
import numpy as np
import tensorrt as trt
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
    def __init__(self, config: ViTConfig = ViTConfig()):
        nn.Module.__init__(self)
        MetaLogger.__init__(self)

        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Convolves over the image extracting patches of size `patch_size * patch_size`.
        # This ensures the output embedding contains `patch_embedding_size` channels.
        self.patcher = nn.Conv2d(
            3, config.patch_embedding_size, config.patch_size, config.patch_size
        )

        # Determine the number of embeddings after the conv layer
        self.sequence_length = (config.image_size // config.patch_size) ** 2

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
        self.sequence_length += 1

        self.encoder = Encoder(
            self.sequence_length,
            config.patch_embedding_size,
            config.encoder_config,
        )

        # Initialise the "head" linear layer
        num_neurons = config.num_classes if config.num_classes > 2 else 1
        self.head = nn.Linear(config.patch_embedding_size, num_neurons)

        self.to(self.device)

    def _check_input_size(self, height: int, width: int):
        if torch.onnx.is_in_onnx_export():
            return

        if height != self.config.image_size or width != self.config.image_size:
            exp = self.config.image_size
            msg = f"Expected (h, w) = ({exp}, {exp}) but got ({height}, {width})."
            self.logger.error(msg)
            raise RuntimeError(msg)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass of the model.

        Parameters
        ----------
        x: `torch.Tensor`
            An input batch of images shaped as `(batch_size, channel, height, width)`.
        """

        # Ensure the input image has the right shape
        h, w = image_batch.size(2), image_batch.size(3)
        self._check_input_size(h, w)

        # Calculate the embeddings for each patch and ensure they form a sequence with
        # shape (batch, sequence_length -1, patch_embedddings_size)
        embed_size = self.config.patch_embedding_size
        seq_len = self.sequence_length - 1
        patch_embeddings: torch.Tensor = self.patcher(image_batch)
        patch_embeddings = patch_embeddings.reshape(-1, embed_size, seq_len)
        patch_embeddings = patch_embeddings.transpose(1, 2)
        patch_embeddings = patch_embeddings.contiguous()

        # Expand the class token so it can be concatenated to each embedding in the
        # batch
        batch_class_token = self.class_token.expand(patch_embeddings.size(0), -1, -1)
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

        # Initialise the model
        model = cls(config)
        model.load_state_dict(model_state["model"])
        model.eval()

        return model

    def export_onnx(
        self, file: Path | str, batch_size: int = 32, verbose: bool = False
    ):
        """
        Converts the PyTorch model into an ONNX and saves it to the specified path.

        Parameters
        ----------
        file:
            The path that the ONNX file will be saved to.

        batch_size:
            The maximum size of a batch the model will process.

        verbose:
            Whether to use verbose logging during ONNX export.
        """
        if "cuda" not in self.device:
            self.logger.warning(
                f"Converting on {self.device}. You must convert on 'cuda' device to "
                "convert to a TensorRT engine."
            )

        # Define a dummy input for the model
        image_size = self.config.image_size
        dummy_input = torch.randn(
            (batch_size, 3, image_size, image_size),
            dtype=torch.float32,
            device=self.device,
        )

        # Export the model to an ONNX file
        self.logger.info("Starting ONNX export...")
        model = self.eval()
        file_str = file.as_posix() if isinstance(file, Path) else file
        torch.onnx.export(
            model,
            dummy_input,
            file_str,
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            verbose=verbose,
        )
        self.logger.info(f"Model exported to {file_str}.")

    def export_tensorrt(self, file: Path | str, batch_size: int = 32):
        """
        Converts the PyTorch model into a TensorRT engine and saves it to the specified
        path.

        Parameters
        ----------
        file:
            The path that the TensorRT engine will be saved to.

        batch_size:
            The maximum size of a batch the model will process.
        """
        # Convert the model to ONNX
        onnx_file = file.with_suffix(".onnx")
        self.export_onnx(onnx_file, batch_size)

        # Create TensorRT tools for converting ONNX to TensorRT
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, trt_logger)

        # Read in the ONNX model and try and parse it
        with open(onnx_file, "rb") as fp:
            onnx_data = fp.read()

        if not parser.parse(onnx_data):
            self.logger.error("Failed to parse the ONNX model.")
            for error in range(parser.num_errors):
                self.logger.error(parser.get_error(error))

        # Display information about input and output shapes
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for input in inputs:
            self.logger.info(f"Model {input.name} shape: {input.shape} {input.dtype}")
        for output in outputs:
            self.logger.info(
                f"Model {output.name} shape: {output.shape} {output.dtype}"
            )

        # Create TensorRT build config
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)
        cache = config.create_timing_cache(b"")
        config.set_timing_cache(cache, ignore_mismatch=False)

        profile = builder.create_optimization_profile()
        im_size = self.config.image_size
        profile.set_shape(
            "input",
            (1, 3, im_size, im_size),
            (batch_size, 3, im_size, im_size),
            (batch_size, 3, im_size, im_size),
        )
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            self.logger.error("Failed to build the TensorRT engine.")
            return

        # Write the serialized engine to a file
        with open(file, "wb") as fp:
            fp.write(serialized_engine)
        self.logger.info(f"Engine saved to {file}")
