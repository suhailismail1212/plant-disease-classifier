"""
config.py — Central place for all project settings.

Instead of scattering magic numbers like "224" and "32" throughout the code,
we define them once here. That way, if we want to experiment (e.g. change
batch size), we change it in one place only.
"""

# ── Dataset ───────────────────────────────────────────────────────────────────

# The Hugging Face dataset identifier — this is the "address" of the dataset
# on the HuggingFace Hub so the datasets library knows where to download from.
DATASET_NAME = "Saon110/bd-crop-vegetable-plant-disease-dataset"

# We are only training on tomato to keep the project manageable.
# The dataset has 13 crops; filtering to one crop reduces training time a lot.
CROP_FILTER = "Tomato"

# Path where we will save the processed (ready-to-train) data locally.
# Using relative paths so this works on any machine.
DATA_DIR = "data/processed"

# ── Model ─────────────────────────────────────────────────────────────────────

# ResNet50 expects images to be exactly 224×224 pixels.
# This is not arbitrary — it's the size the model was originally trained on
# (ImageNet). Feeding the same size keeps the pretrained features meaningful.
IMAGE_SIZE = 224

# Number of colour channels: 3 = Red, Green, Blue.
# Grayscale would be 1 channel, but our plant photos are colour.
NUM_CHANNELS = 3

# ── Training ──────────────────────────────────────────────────────────────────

# Batch size: how many images we feed into the model at once during training.
# 32 is a common default — large enough to be efficient, small enough to fit
# in RAM. Larger batches need more memory; smaller batches mean more updates
# per epoch but noisier gradients.
BATCH_SIZE = 32

# Number of epochs: how many times the model sees the entire training dataset.
# 10 is a reasonable starting point for fine-tuning a pretrained model.
# More epochs = longer training but not always better (risk of overfitting).
NUM_EPOCHS = 10

# Learning rate: controls how big a step the optimiser takes when adjusting
# weights. 0.001 is the standard PyTorch default for Adam.
# Too high → model overshoots. Too low → model learns painfully slowly.
LEARNING_RATE = 0.001

# Train / Validation / Test split ratios.
# 80% of data for training, 10% for validation (tuning), 10% for final test.
TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1

# Random seed: setting a fixed seed means we get the same random shuffles
# every run, so experiments are reproducible.
RANDOM_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────

# Where to save the best model weights during training.
MODEL_SAVE_PATH = "models/best_model.pth"
