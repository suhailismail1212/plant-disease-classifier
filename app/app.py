"""
app.py — A Gradio web interface for the tomato disease classifier.

Gradio is a Python library that turns a plain function into a web page
with almost no extra code — you write a "predict" function, tell Gradio
what the inputs/outputs look like, and it builds the entire UI for you.

Run locally:
    python3 -m app.app

Then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import json
from pathlib import Path

import torch
import gradio as gr
from torchvision import transforms

from src.model import build_model, get_device
from src.config import IMAGE_SIZE, MODEL_SAVE_PATH

# ── Load everything ONCE at startup ────────────────────────────────────────
#
# We load the model and class names here, at module level (not inside the
# predict function), because loading a model from disk is slow. If we did
# it inside predict(), the app would reload the entire model on every
# single image someone uploads — extremely wasteful. Loading once at
# startup means every prediction after that is fast.

device = get_device()

# Load the list of class names in the exact order the model was trained on.
# This file was created back in Session 1's download_data.py.
class_names_path = Path("data/processed/class_names.json")
with open(class_names_path) as f:
    class_names = json.load(f)

# Rebuild the same model architecture we trained, then load the saved weights.
model = build_model(num_classes=len(class_names), freeze_backbone=True)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model = model.to(device)
model.eval()  # inference mode — same reasoning as in evaluate.py

# This must be IDENTICAL to the val/test transform used during training.
# If we preprocessed images differently here, the model would receive
# inputs it was never trained to understand, and predictions would be
# unreliable — even though the model itself is unchanged.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


def predict(image):
    """
    Takes a PIL Image (Gradio passes images in this format by default),
    runs it through the model, and returns a dictionary mapping each
    disease name to its predicted probability.

    Gradio's Label output component automatically turns this dictionary
    into a nice bar-chart-style display of the top predictions.
    """

    if image is None:
        # Gradio calls predict() with None if the user clears the input
        # without uploading anything — handle it gracefully instead of crashing.
        return None

    # Apply the exact same preprocessing used during training/evaluation.
    image_tensor = inference_transform(image)

    # The model expects a BATCH of images, shape (batch_size, 3, 224, 224).
    # We only have one image, shape (3, 224, 224), so we add a fake batch
    # dimension of size 1 using unsqueeze(0). Without this, the model would
    # raise a shape-mismatch error.
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        # Same no_grad() reasoning as evaluation — we're not training,
        # so there's no need to track gradients.
        outputs = model(image_tensor)

        # Softmax converts the model's raw output scores ("logits") into
        # probabilities that sum to 1.0 across all classes. dim=1 means
        # apply softmax across the class dimension (not the batch dimension).
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Build a dictionary like {"Tomato_healthy": 0.97, "Tomato_Early_Blight": 0.02, ...}
    # .item() converts each single-value tensor into a plain Python float,
    # which Gradio's Label component requires.
    confidences = {
        class_names[i]: probabilities[i].item()
        for i in range(len(class_names))
    }

    return confidences


# ── Build the Gradio interface ─────────────────────────────────────────────

# gr.Interface is the simplest way to wrap a function into a full web UI.
# It automatically builds an upload box, a submit button, and a results
# display, based on the input/output types we specify.
demo = gr.Interface(
    fn=predict,

    # gr.Image(type="pil") means the uploaded image arrives in our predict()
    # function as a PIL Image object — exactly what our transform pipeline expects.
    inputs=gr.Image(type="pil", label="Upload a tomato leaf photo"),

    # gr.Label displays a dictionary of {class: probability} as a ranked bar
    # chart. num_top_classes=5 shows only the 5 most likely diseases instead
    # of cluttering the screen with all 9.
    outputs=gr.Label(num_top_classes=5, label="Prediction"),

    title="Tomato Leaf Disease Classifier",
    description=(
        "Upload a photo of a tomato leaf and this model will predict which "
        "of 9 conditions it shows (8 diseases + healthy). "
        "Built with PyTorch, fine-tuned ResNet50, trained on the Bangladesh "
        "Multi-Crop Disease Classification Dataset. Test accuracy: 95.08%."
    ),

    # examples lets users click a sample image instead of finding their own.
    # This list is optional and only works if these example files exist —
    # we will add a few in the next step.
    examples=None,
)


if __name__ == "__main__":
    # launch() starts a local web server. share=False keeps it local-only;
    # we will deploy properly to Hugging Face Spaces separately, which
    # gives a permanent public URL rather than a temporary shareable link.
    demo.launch()
