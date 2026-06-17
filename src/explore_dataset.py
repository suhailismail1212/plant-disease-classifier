"""
explore_dataset.py — A quick script to inspect the dataset BEFORE downloading
all images. Run this first to understand the structure of the data.

It streams only the first few hundred rows so it is fast — no huge download.

Usage:
    python src/explore_dataset.py
"""

from datasets import load_dataset
from collections import Counter
from src.config import DATASET_NAME, CROP_FILTER


def explore():
    print("Streaming first 500 rows from Hugging Face (fast — no full download)...\n")

    # streaming=True means the library fetches rows one by one on demand
    # instead of downloading the entire 19 GB dataset upfront.
    # split="train" means we peek at the training portion only.
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # .take(500) grabs the first 500 rows as an iterator; list() materialises them
    sample = list(dataset.take(500))

    # ── Print the column names ─────────────────────────────────────────────────

    # Columns are the "fields" each row has — like columns in a spreadsheet.
    print("Columns in dataset:", list(sample[0].keys()))

    # Print one example row, skipping the actual image (it's binary data)
    print("\nExample row (image skipped):")
    for key, val in sample[0].items():
        if key != "image":
            print(f"  {key}: {val}")

    # ── Show the label_name distribution ──────────────────────────────────────

    # From the HuggingFace viewer we know the columns are:
    #   image      — the photo
    #   label      — integer index (0, 1, 2 …)
    #   label_name — human-readable string like "Tomato_Healthy"
    #
    # The crop name is the PREFIX of label_name (e.g. "Tomato_", "Banana_")
    # so we filter by checking if label_name starts with our crop name.

    print(f"\nAll unique label_names in this 500-row sample:")
    label_counts = Counter(row["label_name"] for row in sample)
    for name, count in sorted(label_counts.items()):
        print(f"  {name}: {count}")

    # ── Filter to tomato only ──────────────────────────────────────────────────

    tomato_rows = [r for r in sample
                   if str(r["label_name"]).startswith(CROP_FILTER)]

    print(f"\nTomato rows in this 500-row sample: {len(tomato_rows)}")

    if tomato_rows:
        print("Tomato disease classes seen so far:")
        tomato_counts = Counter(r["label_name"] for r in tomato_rows)
        for name, count in sorted(tomato_counts.items()):
            print(f"  {name}: {count}")
    else:
        print("No tomato rows appeared in the first 500 — the dataset may be sorted by crop.")
        print("That is fine; download_data.py filters the full dataset, not just 500 rows.")

    print("\nExploration done. Next step: run download_data.py to fetch everything.")


if __name__ == "__main__":
    explore()
