"""
dataset.py — Defines how PyTorch loads and prepares our images for training.

There are two things in this file:
  1. get_transforms() — a function that defines what to DO to each image
                        before feeding it into the model
  2. get_dataloaders() — a function that creates the actual objects that
                         feed batches of images into training

You will import these in train.py and never need to touch this file again
once it is written.
"""

import json
from pathlib import Path

# torchvision is PyTorch's companion library for images.
# transforms: pre-processing operations applied to images
# datasets:   built-in helpers for loading image folders from disk
# DataLoader: wraps a dataset and serves it in batches
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from src.config import (
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
)


# ── PART 1: Transforms ────────────────────────────────────────────────────────
#
# A "transform" is a function applied to an image before the model sees it.
# We define DIFFERENT transforms for training vs validation/test because:
#
#   Training:   we WANT random variation — flipping, cropping, colour changes.
#               This is called DATA AUGMENTATION. It artificially increases
#               the diversity of what the model sees, so it learns to recognise
#               disease patterns regardless of angle, lighting, or position.
#               Think of it like training a student with many slightly different
#               versions of the same photo.
#
#   Validation/Test: we do NOT want random variation. We need a consistent,
#               deterministic view of each image so the accuracy measurement
#               is fair and reproducible. If we randomly flipped images at
#               test time, the score would vary each run.

def get_transforms():
    """
    Returns a dictionary with two transform pipelines:
      - 'train': augmentation + normalisation
      - 'val':   just resize + normalisation (also used for test)
    """

    # These mean and std values are the ImageNet statistics — the average
    # pixel value and spread across the 1.2 million ImageNet images, per
    # colour channel (Red, Green, Blue).
    #
    # WHY use ImageNet stats? Because ResNet50 was trained with these exact
    # normalisations. When we load its pretrained weights, those weights
    # "expect" inputs that are normalised this way. Using different stats
    # would make the pretrained features meaningless.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        # Compose chains transforms together. Each image passes through
        # them in order, top to bottom.

        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Resize to 224×224. We already did this when saving, but we
        # include it here as a safety net in case of any size mismatch.

        transforms.RandomHorizontalFlip(p=0.5),
        # With 50% probability, flip the image left-to-right.
        # A diseased leaf looks the same whether it is facing left or right.
        # Flipping doubles the variety of orientations the model sees.

        transforms.RandomRotation(degrees=15),
        # Randomly rotate up to 15 degrees either way.
        # A phone photo of a leaf could be taken at any angle.
        # Training on rotated images makes the model rotation-tolerant.

        transforms.ColorJitter(
            brightness=0.3,   # randomly darken or brighten by up to 30%
            contrast=0.3,     # randomly increase or decrease contrast
            saturation=0.3,   # randomly change colour saturation
        ),
        # Real-world photos vary in lighting conditions — shade, direct sun,
        # indoor fluorescent light. ColorJitter teaches the model to focus
        # on disease SHAPE and PATTERN rather than exact colour.

        transforms.RandomCrop(IMAGE_SIZE, padding=10),
        # Add 10 pixels of black border then randomly crop back to 224×224.
        # This means the disease symptom might appear slightly off-centre.
        # It teaches the model that the symptom can be anywhere in the image.

        transforms.ToTensor(),
        # Convert the PIL Image (pixel values 0-255) to a PyTorch Tensor
        # (float values 0.0-1.0). This is required — PyTorch cannot process
        # PIL Images directly, only Tensors.
        # Shape changes from (H, W, C) to (C, H, W) — PyTorch puts channels first.

        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        # Subtract the mean and divide by the std for each colour channel.
        # This centres the pixel values around 0 (roughly -2 to +2 range).
        # Neural networks train faster and more stably on normalised inputs
        # because the gradient magnitudes stay balanced across channels.
        # MUST be last — it operates on Tensors, not PIL Images.
    ])

    val_transform = transforms.Compose([
        # No augmentation here — just the minimum needed to feed the model.

        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Resize to the required input size.

        transforms.ToTensor(),
        # Convert to Tensor.

        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        # Same normalisation as training — the model was updated using
        # these stats, so evaluation must use them too.
    ])

    return {
        "train": train_transform,
        "val":   val_transform,
        "test":  val_transform,   # test uses identical transform to val
    }


# ── PART 2: DataLoaders ───────────────────────────────────────────────────────
#
# A DataLoader is the object that actually serves images to the training loop.
# It handles:
#   - Reading images from disk (using ImageFolder, which reads the folder structure)
#   - Applying transforms
#   - Grouping images into batches of BATCH_SIZE
#   - Shuffling the training data each epoch
#   - Loading images in parallel using multiple CPU workers

def get_dataloaders():
    """
    Creates and returns three DataLoaders: train, val, and test.

    Also returns the list of class names so training and the app know
    which index maps to which disease name.
    """

    transforms_dict = get_transforms()
    base_path = Path(DATA_DIR)

    # ImageFolder is a PyTorch built-in that reads a folder structured as:
    #   root/class_a/img1.jpg
    #   root/class_b/img2.jpg
    # It automatically assigns integer labels based on alphabetical folder order.
    # Our data/processed/train/ folder is already in exactly this format.
    datasets_dict = {
        split: datasets.ImageFolder(
            root=str(base_path / split),       # path to train/, val/, or test/
            transform=transforms_dict[split],  # which transforms to apply
        )
        for split in ["train", "val", "test"]
    }

    # Load the class names we saved during data preparation.
    # ImageFolder also creates its own class list (dataset.classes), but we
    # load from JSON to guarantee the same order as training — important for
    # the Gradio app later when it needs to map index → disease name.
    with open(base_path / "class_names.json") as f:
        class_names = json.load(f)

    # Build a DataLoader for each split
    dataloaders = {}

    for split, dataset in datasets_dict.items():

        is_train = (split == "train")

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            # shuffle=True only for training. Shuffling means each epoch the
            # model sees images in a different order — prevents it from learning
            # the ORDER of images rather than their content.
            # Val and test are NOT shuffled so results are reproducible.
            shuffle=is_train,
            # num_workers=2 means 2 background processes load images from disk
            # in parallel while the GPU/CPU is busy processing the current batch.
            # Without this, the model sits idle waiting for disk reads.
            # 2 is safe on a MacBook; on a server you'd use 4 or 8.
            num_workers=2,
            # pin_memory=True speeds up CPU→GPU transfer if you have a GPU.
            # On a MacBook CPU it has no effect but doesn't hurt.
            pin_memory=False,
        )

    # Print a summary so we can verify the numbers look right
    print("DataLoaders ready:")
    for split, dataset in datasets_dict.items():
        print(f"  {split:5s}: {len(dataset):5d} images, "
              f"{len(dataloaders[split]):3d} batches of {BATCH_SIZE}")

    print(f"\nClasses ({len(class_names)}):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    return dataloaders, class_names
