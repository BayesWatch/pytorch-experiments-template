import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
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

    nn.init.xavier_uniform(patch_attention_mask)

    optimizer = optimizer(params=[patch_attention_mask], lr=0.1, weight_decay=0.0)
    model = model.eval()

    inputs, targets = inputs.to(device), targets.to(device)

    output_images = []

    for iter in range(num_iter):
        patch_attention_mask_input_size_match = F.adaptive_avg_pool2d(
            input=patch_attention_mask.to(device),
            output_size=(inputs.shape[-2], inputs.shape[-1]),
        )

        inputs_masked = inputs * patch_attention_mask_input_size_match.sigmoid()
        logits = model(inputs_masked)

        print(patch_attention_mask.sigmoid().sum(),
              criterion(input=logits, target=targets))

        loss = (
            importance_classification_loss * criterion(input=logits, target=targets)
            + importance_mask_size
            * patch_attention_mask_input_size_match.sigmoid().sum()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 50 == 0:
            mixed_results = torch.cat(
                [patch_attention_mask_input_size_match.sigmoid()], dim=0
            )
            output_images.extend(mixed_results)

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

    reconstruct_original_transforms = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        ]
    )

    input_tensor = preprocess(input_image)
    inputs = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    originals = reconstruct_original_transforms(input_image).unsqueeze(0)
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
        patch_size=(32, 32),
        device=torch.cuda.current_device(),
        criterion=F.cross_entropy,
        optimizer=optim.Adam,
        num_iter=1000,
        importance_mask_size=1,
        importance_classification_loss=100000,
    )

    output_masks = torch.stack(outputs, dim=0)

    print(output_masks.shape, originals.shape)

    originals = originals.repeat([output_masks.shape[0], 1, 1, 1]).to(
        torch.cuda.current_device()
    )

    masked_originals = output_masks * originals

    masked_originals_images = torch.cat(
        torch.unbind(masked_originals, dim=0), dim=1
    )  # .permute([1, 2, 0])

    print(masked_originals_images.shape)

    torchvision.utils.save_image(
        tensor=masked_originals_images, fp="masked_image_dog.png"
    )
