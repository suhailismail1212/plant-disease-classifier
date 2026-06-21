"""
test_dataloader.py — Sanity check for the data pipeline.

Run this to confirm images load correctly, shapes are right, and
the DataLoader serves batches as expected. Takes about 10 seconds.

Usage:
    python3 -m src.test_dataloader
"""

import torch
from src.dataset import get_dataloaders


def test():
    print("Building DataLoaders...\n")
    dataloaders, class_names = get_dataloaders()

    # Grab exactly one batch from the training loader
    images, labels = next(iter(dataloaders["train"]))

    # images is a Tensor of shape (batch_size, channels, height, width)
    # labels is a Tensor of shape (batch_size,) containing integer class indices
    print(f"\nOne training batch:")
    print(f"  images shape : {images.shape}")
    # Expected: torch.Size([32, 3, 224, 224])
    #   32  = batch size
    #   3   = RGB channels
    #   224 = height
    #   224 = width

    print(f"  labels shape : {labels.shape}")
    # Expected: torch.Size([32])

    print(f"  pixel range  : min={images.min():.3f}, max={images.max():.3f}")
    # After normalisation, values should be roughly -2.5 to +2.5 (not 0-255)

    print(f"\nFirst 8 labels in this batch:")
    for i in range(8):
        label_idx = labels[i].item()   # .item() converts a 1-element Tensor to a plain Python int
        print(f"  label {label_idx} → {class_names[label_idx]}")

    print("\nAll checks passed. DataLoader is working correctly.")


if __name__ == "__main__":
    test()
