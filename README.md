# ViT-TensorRT
A re-implementation of the original Vision Transformer (ViT) model in PyTorch. This
repository also makes it easy to productionize the model with ONNX and TensorRT export.

## Installation
ViT-TensorRT requires your system to have a NVIDIA GPU with CUDA installed. CUDA `12.4`
has been tested with this repository.

To install `vit-tensorrt` and its dependencies, run:

```bash
pip install vit-tensorrt
```

## Training
Training the model can be achieved in a few lines of code:

```python
from pathlib import Path
from vit_tensorrt.config import TrainConfig
from vit_tensorrt.model import ViT

model = ViT()
model.fit(TrainConfig(data_path=Path("path/to/data")))
```

This assumes the data is stored in a directory with the following structure:

```
├── data
    ├── train
        ├── images
            ├── uuid1.jpg
            ├── uuid2.jpg
            └── ...
        └── labels
            ├── uuid1.txt
            ├── uuid2.txt
            └── ...
    ├── val/
        └── ...
    ├── test/
        └── ...
```

Where the label file contains a single number indicating the class of the corresponding
image.

## Export

### ONNX export
Exporting the model to ONNX can be done with:

```python
model.export_onnx("path/to/output.onnx")
```

The model will be exported with ONNX opset version `18`.

### TensorRT export
Exporting the model to a TensorRT engine can be done with:

```python
model.export_tensorrt("path/to/output.onnx")
```

The model will be exported with TensorRT version `10.4.0`.

## Deploy with Triton Inference Server
TensorRT engines are exported using TensoRT version `10.4.0` therefore any Triton
Inference Servers compatible with this version of TensorRT can be used. The deployment
of ViT-TensorRT has been tested with `nvcr.io/nvidia/tritonserver:24.09-py3`. If you
want to investigate alternatives, please refer to [Triton's Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags).

The easiest way to deploy the model is by running the following in your terminal:

```bash
nvidia-docker run \
    -dit \
    --net=host \
    --name vit-triton \
    -v <path-to-model>:/models/vit/1 \
    nvcr.io/nvidia/tritonserver:24.09-py3 \
    tritonserver --model-repository=/models
```

The above assumes you have renamed your model to `model.plan`. If you want to configure
how your model runs in Triton, please refer to [Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).

### Triton inference from Python client
To perform inference on the deployed model, you can use Python's Triton Inference
Client:

```python
from tritonclient import http

client = http.InferenceServerClient("localhost:8000")

# Create input data
inputs = [http.InferInput("input", [32, 3, 256, 256], "FP32")]
inputs[0].set_data_from_numpy(np.random.rand(32, 3, 256, 256).astype(np.float32))

# Run inference
results = client.infer("vit", inputs)
output = results.as_numpy("output")
```

Where `input` and `output` are the names of the input and output layers of the model,
respectively, and `vit` is the name of the model in Triton. Make sure the input size you
specify matches the size that you trained the model with.

To increase model throughput, calls to Triton Inference Server should be made with
shared GPU memory. This is more complicated to setup, but if you are interested please
raise an issue on the repository and an example can be provided.