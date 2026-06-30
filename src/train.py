"""
train.py — Trains the ResNet50 model on tomato disease images.

This is the script you actually run to train the model. It:
  1. Loads the data (via dataset.py)
  2. Builds the model (via model.py)
  3. Loops over the training data for several epochs
  4. After each epoch, checks accuracy on validation data
  5. Saves the best-performing model to disk

Usage:
    python3 -m src.train
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import build_model, get_device
from src.config import NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs ONE full pass over the training data.

    Returns the average loss and accuracy for this epoch.
    """

    # model.train() puts the model in "training mode". This matters because
    # some layers (like Dropout, BatchNorm) behave differently during
    # training vs evaluation. ResNet50 uses BatchNorm internally, so this
    # call is necessary even though we don't see those layers directly.
    model.train()

    running_loss = 0.0       # accumulates total loss across all batches
    correct_predictions = 0  # counts how many images were classified correctly
    total_images = 0         # counts total images processed

    for images, labels in dataloader:
        # Move this batch of images and labels to the same device the model
        # is on (MPS/CUDA/CPU). Tensors must be on the same device to
        # interact — you can't multiply a GPU tensor with a CPU tensor.
        images, labels = images.to(device), labels.to(device)

        # ── Forward pass ────────────────────────────────────────────────────
        # Zero out gradients from the previous batch. PyTorch accumulates
        # gradients by default, so we must clear them before each new batch
        # or they would incorrectly add up across batches.
        optimizer.zero_grad()

        # Run the images through the model to get predictions.
        # outputs shape: (batch_size, num_classes) — one score per class
        # per image. These are "logits" — raw, unnormalised scores.
        outputs = model(images)

        # Calculate how wrong the predictions are using the loss function.
        # criterion (CrossEntropyLoss) compares the predicted scores against
        # the true labels and produces a single number: lower = better.
        loss = criterion(outputs, labels)

        # ── Backward pass ───────────────────────────────────────────────────
        # backward() calculates the gradient of the loss with respect to
        # every trainable parameter — i.e. "how much would changing this
        # weight slightly change the loss?" This uses calculus
        # (backpropagation) automatically — we never write derivatives by hand.
        loss.backward()

        # step() actually updates the model's weights using the gradients
        # just calculated, scaled by the learning rate. This is the moment
        # the model "learns" from this batch.
        optimizer.step()

        # ── Track statistics ────────────────────────────────────────────────
        # loss.item() converts the loss tensor to a plain Python float.
        # We multiply by images.size(0) (the batch size) because loss is
        # already averaged per batch — we want the total, then we'll
        # re-average over the whole epoch at the end.
        running_loss += loss.item() * images.size(0)

        # torch.max returns (values, indices). We want the index of the
        # highest score for each image — that's the model's predicted class.
        _, predicted = torch.max(outputs, dim=1)

        # Compare predictions to true labels element-wise, sum up the matches.
        correct_predictions += (predicted == labels).sum().item()
        total_images += labels.size(0)

    epoch_loss = running_loss / total_images
    epoch_accuracy = correct_predictions / total_images
    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set. Same logic as training but
    WITHOUT updating any weights — we are only measuring, not learning.
    """

    # model.eval() switches BatchNorm/Dropout layers into evaluation mode.
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    # torch.no_grad() tells PyTorch not to track gradients for anything
    # inside this block. We don't need gradients during validation since
    # we're not calling backward() — this saves memory and speeds things up.
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_images += labels.size(0)

    epoch_loss = running_loss / total_images
    epoch_accuracy = correct_predictions / total_images
    return epoch_loss, epoch_accuracy


def train_model():
    """
    Main training function — orchestrates the full training process across
    all epochs and saves the best model.
    """

    device = get_device()

    print("\nLoading data...")
    dataloaders, class_names = get_dataloaders()

    print("\nBuilding model...")
    model = build_model(num_classes=len(class_names), freeze_backbone=True)
    model = model.to(device)  # move all model weights onto the chosen device

    # ── Loss function ───────────────────────────────────────────────────────
    # CrossEntropyLoss is the standard loss function for multi-class
    # classification. It combines a Softmax (turns raw scores into
    # probabilities) and a Negative Log-Likelihood loss (penalises confident
    # wrong answers more than unsure wrong answers) into one efficient step.
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────────────────────────
    # The optimizer is the algorithm that updates model weights based on
    # gradients. We use Adam — it adapts the learning rate per-parameter
    # automatically, which generally converges faster and more reliably
    # than plain Stochastic Gradient Descent (SGD), especially for
    # fine-tuning pretrained models.
    #
    # filter(lambda p: p.requires_grad, ...) passes ONLY the trainable
    # parameters (our new final layer) to the optimizer. There's no point
    # giving it the frozen layers — it would waste memory tracking
    # momentum/state for parameters that never change.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0  # tracks the best validation accuracy seen so far

    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )

        val_loss, val_acc = validate_one_epoch(
            model, dataloaders["val"], criterion, device
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc*100:.2f}% | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc*100:.2f}% | "
            f"{elapsed:.1f}s"
        )

        # ── Save the model only if it's the best one so far ───────────────────
        # We track validation accuracy (not training accuracy) because
        # training accuracy can be misleadingly high if the model is
        # overfitting — memorising training data rather than generalising.
        # Validation accuracy tells us how well the model performs on
        # images it has never directly learned from.
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  → New best model saved (val acc: {val_acc*100:.2f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy*100:.2f}%")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
