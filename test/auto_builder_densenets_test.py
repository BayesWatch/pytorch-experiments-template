from models.auto_builder_densenet import AutoDenseNet
from models.auto_builder_models import (
    EasyPeasyResNet,
    EasyPeasyConvNet,
    EasyPeasyConvRelationalNet,
)

import torch

from models.auto_builder_transformers import (
    EasyPeasyViTFlatten,
    EasyPeasyViTLastTimeStep,
)

RUN_CUDA_TESTS = False


def apply_to_test_device(model, input_tensor):
    if torch.cuda.is_available() and RUN_CUDA_TESTS:
        model = model.to(torch.cuda.current_device())

        input_tensor = input_tensor.to(torch.cuda.current_device())

    else:

        model = model.to(torch.device("cpu"))

        input_tensor = input_tensor.to(torch.device("cpu"))

    return model, input_tensor

def test_AutoDenseNet_layer_output_shape():
    model = AutoDenseNet(
        num_classes=10,
        num_filters=16,
        num_stages=3,
        num_blocks=4,
        dilated=False
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10

def test_AutoDenseNetDilated_layer_output_shape():
    model = AutoDenseNet(
        num_classes=10,
        num_filters=16,
        num_stages=3,
        num_blocks=4,
        dilated=True
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10