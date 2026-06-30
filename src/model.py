"""
model.py — Defines our ResNet50-based model for tomato disease classification.

This file has one function: build_model(). It loads ResNet50 pretrained on
ImageNet, then modifies it so it outputs 9 tomato disease classes instead
of ImageNet's original 1000 classes.
"""

import torch
import torch.nn as nn                  # nn = "neural network" — contains layer building blocks
from torchvision import models         # Pretrained model architectures live here


def build_model(num_classes, freeze_backbone=True):
    """
    Builds a ResNet50 model adapted for our number of classes.

    Args:
        num_classes:     how many output classes we need (9, for tomato diseases)
        freeze_backbone: if True, freeze all pretrained layers except the new
                          final layer. If False, all layers are trainable.

    Returns:
        A PyTorch model ready for training.
    """

    # ── Load ResNet50 with pretrained ImageNet weights ────────────────────────

    # weights="IMAGENET1K_V2" downloads (or loads from cache) the weights
    # learned from training on 1.2 million ImageNet images. These weights
    # already encode useful general-purpose visual features: edges, textures,
    # colour gradients, shapes — all useful for recognising leaf diseases too.
    model = models.resnet50(weights="IMAGENET1K_V2")

    # ── Freeze the pretrained layers ───────────────────────────────────────────

    if freeze_backbone:
        # "Freezing" a layer means telling PyTorch not to update its weights
        # during training. requires_grad = False stops PyTorch from computing
        # gradients for that parameter, which also makes training faster
        # since there's less math to do.
        #
        # WHY freeze the backbone? With only ~11,000 training images, trying
        # to update all 25 million of ResNet50's parameters would very likely
        # overfit — the model would memorise our small dataset rather than
        # learning generalisable disease patterns. By freezing the pretrained
        # feature-extraction layers, we keep their general visual knowledge
        # intact and only train the small new layer we add for our task.
        for param in model.parameters():
            param.requires_grad = False

    # ── Replace the final classification layer ────────────────────────────────

    # ResNet50's final layer is called "fc" (fully connected). In the original
    # model it maps from 2048 features (ResNet50's internal feature size) to
    # 1000 outputs (ImageNet's 1000 object classes).
    #
    # model.fc.in_features reads the input size of that layer (2048) so we
    # don't have to hard-code it — this makes the code robust if we ever
    # swap to a different ResNet variant.
    num_features = model.fc.in_features

    # We replace it with a new Linear layer that maps from the same 2048
    # features to OUR number of classes (9). This new layer starts with
    # random weights and is the ONLY part of the network that gets trained
    # from scratch — everything before it is reused, pretrained knowledge.
    #
    # Because this new layer is created fresh (not loaded from pretrained
    # weights), its requires_grad is True by default — so even with
    # freeze_backbone=True, this layer WILL be trained. That's the point:
    # we want exactly this layer, and only this layer, to learn.
    model.fc = nn.Linear(num_features, num_classes)

    return model


def get_device():
    """
    Detects the best available hardware to run on.

    Returns a torch.device that PyTorch uses to know where to put tensors
    and run computations.
    """

    if torch.backends.mps.is_available():
        # MPS = Metal Performance Shaders — Apple's GPU acceleration framework
        # for Mac. If you have an M1/M2/M3 Mac, this uses the GPU cores,
        # which is significantly faster than the CPU for training.
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # CUDA = NVIDIA's GPU framework. Not relevant on Mac, but included
        # so this code also works correctly on a Windows/Linux machine with
        # an NVIDIA GPU (e.g. if you ever train on Google Colab).
        device = torch.device("cuda")
    else:
        # Fallback: plain CPU. Slower, but works everywhere.
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


if __name__ == "__main__":
    # Quick manual test: build the model and print a summary of trainable
    # vs frozen parameters. Run with: python3 -m src.model
    model = build_model(num_classes=9)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
