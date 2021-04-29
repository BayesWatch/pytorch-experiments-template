import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np


def train_attention_masks(
    inputs,
    targets,
    model,
    patch_size,
    device,
    criterion,
    optimizer,
    num_iter,
    importance_classification_loss,
    importance_mask_size,
):
    total_patches = int(np.floor(inputs.shape[-1] / patch_size[-1]))
    patch_attention_mask = nn.Parameter(
        data=torch.empty(
            size=(inputs.shape[0], inputs.shape[1], total_patches, total_patches)
        ),
        requires_grad=True,
    )

    optimizer = optimizer(params=[patch_attention_mask])
    model = model.eval()

    inputs, targets = inputs.to(device), targets.to(device)

    output_images = []

    for _ in range(num_iter):
        patch_attention_mask_input_size_match = F.adaptive_avg_pool2d(
            input=patch_attention_mask.to(device),
            output_size=(inputs.shape[-2], inputs.shape[-1]),
        )

        inputs_masked = inputs * patch_attention_mask_input_size_match.sigmoid()
        logits = model(inputs_masked)

        loss = (
            importance_classification_loss * criterion(input=logits, target=targets)
            + importance_mask_size
            * patch_attention_mask_input_size_match.sigmoid().sum()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mixed_results = torch.cat([inputs_masked, inputs], dim=0)
        output_images.append(mixed_results)

    return output_images


if __name__ == "__main__":
    import urllib

    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    from PIL import Image
    from torchvision import transforms

    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(input_image)
    inputs = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    targets = torch.Tensor([463]).long()

    model = torch.hub.load(
        "pytorch/vision:v0.6.0",
        "resnet18",
        pretrained=True,
    )

    model = model.to(torch.cuda.current_device())

    outputs = train_attention_masks(
        inputs,
        targets,
        model,
        patch_size=(3, 3),
        device=torch.cuda.current_device(),
        criterion=F.cross_entropy,
        optimizer=optim.Adam,
        num_iter=100, importance_mask_size=0.5, importance_classification_loss=0.5
    )

    outputs = torch.stack(outputs, dim=0)
    print(outputs.shape)
