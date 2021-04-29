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


def test_EasyPeasyViTFlatten_layer_output_shape():
    model = EasyPeasyViTFlatten(
        grid_patch_size=32,
        transformer_num_filters=512,
        transformer_num_layers=8,
        transformer_num_heads=8,
        model_name_to_download=None,
        pretrained=False,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 128, 128))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_EasyPeasyViTLastTimeStep_layer_output_shape():
    model = EasyPeasyViTLastTimeStep(
        grid_patch_size=32,
        transformer_num_filters=512,
        transformer_num_layers=8,
        transformer_num_heads=8,
        model_name_to_download=None,
        pretrained=False,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 128, 128))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_EasyPeasyViTFlattenPretrained_layer_output_shape():
    model = EasyPeasyViTFlatten(
        grid_patch_size=32,
        transformer_num_filters=768,
        transformer_num_layers=12,
        transformer_num_heads=12,
        model_name_to_download="ViT-B-32",
        pretrained=True,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_EasyPeasyViTLastTimeStepPretrained_layer_output_shape():
    model = EasyPeasyViTFlatten(
        grid_patch_size=32,
        transformer_num_filters=768,
        transformer_num_layers=12,
        transformer_num_heads=12,
        model_name_to_download="ViT-B-32",
        pretrained=True,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10

def test_AutoDenseNet_layer_output_shape():
    model = AutoDenseNet(
        num_classes=10,
        num_filters=16,
        num_stages=3,
        num_blocks=3,
        dilated=False
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10