"""
evaluate.py — Evaluates the trained model on the test set and produces:
  1. Overall test accuracy
  2. Per-class accuracy (which diseases does the model struggle with?)
  3. A confusion matrix plot saved to models/confusion_matrix.png
  4. A training curve plot saved to models/training_curves.png

Run this AFTER training is complete.

Usage:
    python3 -m src.evaluate
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from src.dataset import get_dataloaders
from src.model import build_model, get_device
from src.config import MODEL_SAVE_PATH


def evaluate_model():
    device = get_device()

    print("\nLoading data...")
    dataloaders, class_names = get_dataloaders()

    print("Loading trained model weights...")
    model = build_model(num_classes=len(class_names), freeze_backbone=True)

    # load_state_dict loads the saved weight values back into the model
    # architecture. map_location=device ensures the weights are moved to
    # whatever device we're using (MPS/CPU) even if they were saved on a
    # different device.
    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=device)
    )
    model = model.to(device)

    # model.eval() switches BatchNorm to evaluation mode — same as in train.py
    model.eval()

    # ── Collect predictions on the test set ───────────────────────────────────

    all_labels      = []   # true class labels for every test image
    all_predictions = []   # model's predicted class for every test image

    print("Running model on test set...")
    with torch.no_grad():
        for images, labels in dataloaders["test"]:
            images = images.to(device)

            outputs = model(images)

            # dim=1 means "take the max across the class dimension"
            # i.e. for each image, find which class has the highest score
            _, predicted = torch.max(outputs, dim=1)

            # .cpu() moves tensor back to CPU; .tolist() converts to Python list
            all_labels.extend(labels.tolist())
            all_predictions.extend(predicted.cpu().tolist())

    # Convert to numpy arrays — scikit-learn functions expect numpy, not lists
    all_labels      = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # ── Overall accuracy ──────────────────────────────────────────────────────

    overall_accuracy = (all_labels == all_predictions).mean() * 100
    print(f"\nTest Accuracy: {overall_accuracy:.2f}%")

    # ── Per-class accuracy and report ─────────────────────────────────────────

    # classification_report prints precision, recall, and F1 per class.
    # precision: of all images the model CALLED "Blight", how many actually were?
    # recall:    of all actual "Blight" images, how many did the model find?
    # F1:        harmonic mean of precision and recall — overall class performance
    print("\nPer-class report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=3,
    ))

    # ── Confusion matrix ──────────────────────────────────────────────────────

    # A confusion matrix is a grid where:
    #   rows = true class
    #   columns = predicted class
    #   cell [i][j] = how many images of class i were predicted as class j
    #
    # A perfect model has large numbers on the diagonal and zeros everywhere
    # else. Off-diagonal values show which classes get confused with which.
    cm = confusion_matrix(all_labels, all_predictions)

    # Normalise by row so each cell shows a PERCENTAGE of that true class
    # rather than raw counts (makes different-sized classes comparable)
    cm_normalised = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Short display names — strip the "Tomato_" prefix to fit in the plot
    short_names = [n.replace("Tomato_", "").replace("_", "\n") for n in class_names]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalised, cmap="Blues", vmin=0, vmax=1)

    # Add colour bar legend
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion of true class", fontsize=11)

    # Label axes
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title("Confusion Matrix — Tomato Disease Classifier", fontsize=14)

    # Write the percentage value inside each cell
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_normalised[i, j]
            # Use white text on dark cells so it stays readable
            text_colour = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=8, color=text_colour)

    plt.tight_layout()
    save_path = Path("models/confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nConfusion matrix saved to {save_path}")
    plt.close()


def plot_training_curves(history: dict):
    """
    Plots training and validation loss/accuracy curves from training history.

    history should be a dict with keys:
        train_loss, val_loss, train_acc, val_acc
    Each key maps to a list of values — one per epoch.

    Call this from train.py after training finishes, passing the recorded
    per-epoch metrics.
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss curve ────────────────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val loss",   markersize=4)
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy curve ────────────────────────────────────────────────────────
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]

    ax2.plot(epochs, train_acc_pct, "b-o", label="Train accuracy", markersize=4)
    ax2.plot(epochs, val_acc_pct,   "r-o", label="Val accuracy",   markersize=4)
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Tomato Disease Classifier", fontsize=14)
    plt.tight_layout()

    save_path = Path("models/training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()

    # Also save the raw history numbers as JSON so we can re-plot later
    history_path = Path("models/training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    evaluate_model()
