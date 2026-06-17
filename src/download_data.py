"""
download_data.py — Downloads the tomato-only subset of the dataset from
Hugging Face and saves the image files to data/processed/.

The dataset columns are:
  image       — PIL Image object
  label       — integer class index (0, 1, 2 …)
  label_name  — string like "Tomato_Healthy" or "Banana_Black_Pitting_or_Banana_Rust"

We filter to rows where label_name starts with "Tomato" and use label_name as
the folder name, which is exactly what PyTorch's ImageFolder expects.

Run this once. After it finishes, training works offline.

Usage:
    python src/download_data.py
"""

import json                        # For saving the class-name list as a JSON file
from pathlib import Path           # Nicer cross-platform path handling
from datasets import load_dataset, concatenate_datasets
from PIL import Image              # Pillow: saves images to disk
from tqdm import tqdm              # Prints a live progress bar
from src.config import (
    DATASET_NAME,
    CROP_FILTER,
    DATA_DIR,
    IMAGE_SIZE,
    RANDOM_SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


def download_and_prepare():
    """
    Downloads the full dataset, filters to tomato images, shuffles,
    splits 80/10/10, resizes to 224×224, and saves as JPEGs on disk.
    """

    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face...")
    print("The first run downloads up to ~19 GB — this will take a while.\n")

    # load_dataset fetches all splits and caches them in ~/.cache/huggingface/
    # On repeat runs it reads from cache instead of re-downloading.
    # trust_remote_code=True is required for datasets that ship a custom
    # loading script — only use it for datasets you trust.
    dataset = load_dataset(DATASET_NAME)

    # ── Merge all splits so we can re-split ourselves ─────────────────────────

    # The dataset already has train/validation/test splits, but we want to
    # control the ratios ourselves, so we combine everything then re-split.
    print("Merging all provided splits...")

    all_data = dataset["train"]
    if "validation" in dataset:
        all_data = concatenate_datasets([all_data, dataset["validation"]])
    if "test" in dataset:
        all_data = concatenate_datasets([all_data, dataset["test"]])

    print(f"Total rows after merging: {len(all_data)}")

    # ── Filter to tomato images only ──────────────────────────────────────────

    # The crop name is the prefix of label_name, e.g. "Tomato_Healthy".
    # str.startswith() checks exactly that prefix — fast and reliable.
    print(f"\nFiltering to rows where label_name starts with '{CROP_FILTER}'...")

    tomato_data = all_data.filter(
        lambda row: str(row["label_name"]).startswith(CROP_FILTER)
    )

    print(f"Found {len(tomato_data)} tomato images in total.")

    # ── Discover class names ───────────────────────────────────────────────────

    # Collect every unique label_name that appears in the tomato subset.
    # sorted() guarantees alphabetical order so class index 0 is always the
    # same disease across different runs — reproducibility matters.
    class_names = sorted(set(tomato_data["label_name"]))

    print(f"\nFound {len(class_names)} tomato disease classes:")
    for i, name in enumerate(class_names):
        print(f"  {i:2d}: {name}")

    # ── Shuffle then split 80 / 10 / 10 ──────────────────────────────────────

    # Shuffle first so each split is a random sample of all classes,
    # not just the first (or last) chunk of the sorted dataset.
    tomato_data = tomato_data.shuffle(seed=RANDOM_SEED)

    total     = len(tomato_data)
    train_end = int(total * TRAIN_SPLIT)              # e.g. 80 % of total
    val_end   = train_end + int(total * VAL_SPLIT)    # next 10 %

    # .select(range(a, b)) picks rows by index — like list slicing
    train_data = tomato_data.select(range(0,         train_end))
    val_data   = tomato_data.select(range(train_end, val_end))
    test_data  = tomato_data.select(range(val_end,   total))

    print(f"\nSplit sizes → train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    # ── Save images to disk in ImageFolder format ─────────────────────────────

    # PyTorch's ImageFolder loader expects this exact structure:
    #
    #   data/processed/
    #     train/
    #       Tomato_Healthy/
    #         train_000001.jpg
    #       Tomato_Bacterial_Blight/
    #         train_000002.jpg
    #     val/
    #       ...
    #     test/
    #       ...
    #
    # Each sub-folder name automatically becomes the class label when we
    # later call datasets.ImageFolder("data/processed/train").

    base_path = Path(DATA_DIR)

    for split_name, split_ds in [("train", train_data), ("val", val_data), ("test", test_data)]:
        print(f"\nSaving {split_name} images ({len(split_ds)} total)...")

        for i, row in enumerate(tqdm(split_ds, desc=split_name)):

            label_name = row["label_name"]   # e.g. "Tomato_Healthy"

            # Build the output folder path and create it if needed
            # parents=True creates any missing parent directories
            # exist_ok=True means no error if the folder already exists
            folder = base_path / split_name / label_name
            folder.mkdir(parents=True, exist_ok=True)

            # row["image"] is already a PIL Image object (HuggingFace decodes it)
            img = row["image"]

            # Resize to 224×224 — the size ResNet50 was designed for
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

            # Convert to RGB (3 channels) in case any image is RGBA or grayscale
            img = img.convert("RGB")

            # f"{i:06d}" pads the number to 6 digits, e.g. 000001, 000042
            # This keeps files sorted correctly in any file browser
            img.save(folder / f"{split_name}_{i:06d}.jpg")

    # ── Save the class-name list ───────────────────────────────────────────────

    # We save class_names as a JSON file so the training script and the Gradio
    # app can load it without re-scanning the folder tree every time.
    class_file = base_path / "class_names.json"
    with open(class_file, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nClass names saved → {class_file}")
    print("Data preparation complete. You can now run training.")


if __name__ == "__main__":
    download_and_prepare()
