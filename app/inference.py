from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageOps


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "model.onnx"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = 300

NORM_MEAN = np.array(
    [0.76264286, 0.54455656, 0.5684541],
    dtype=np.float32,
)

NORM_STD = np.array(
    [0.14133665, 0.15278324, 0.1704188],
    dtype=np.float32,
)


class_names: list[str] = [
    "akiec",
    "bcc",
    "bkl",
    "nv",
    "mel",
]

class_labels: dict[str, str] = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
}


def load_class_metadata() -> None:
    """
    Load class names and labels from class_names.json.
    Falls back to the known classes above if loading fails.
    """
    global class_names, class_labels

    if not CLASS_NAMES_PATH.exists():
        return

    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        loaded_names = metadata.get("class_names")
        loaded_labels = metadata.get("class_labels")

        if isinstance(loaded_names, list) and len(loaded_names) == 5:
            class_names = loaded_names

        if isinstance(loaded_labels, dict):
            class_labels = {
                key: str(value).strip()
                for key, value in loaded_labels.items()
            }

    except Exception as exc:
        print(f"[WARNING] Could not load class metadata: {exc}")


load_class_metadata()


def create_onnx_session() -> ort.InferenceSession:
    """
    Create the ONNX inference session.

    CPUExecutionProvider is intentionally used for Railway's normal
    CPU deployment.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}"
        )

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    return session


SESSION = create_onnx_session()

INPUT_NAME = SESSION.get_inputs()[0].name
OUTPUT_NAMES = [output.name for output in SESSION.get_outputs()]

print("[INFO] ONNX model loaded")
print(f"[INFO] Input name: {INPUT_NAME}")
print(f"[INFO] Output names: {OUTPUT_NAMES}")
print(
    f"[INFO] Providers: "
    f"{SESSION.get_providers()}"
)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Apply the exact validation preprocessing used during training:

    Resize 300x300
    RGB
    ToTensor equivalent
    Normalize(mean, std)

    Returns:
        numpy array with shape (1, 3, 300, 300)
    """
    image = image.convert("RGB")
    image = image.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    # HWC -> CHW
    image_array = np.transpose(image_array, (2, 0, 1))

    # Normalize exactly as PyTorch transforms.Normalize
    image_array = (
        image_array - NORM_MEAN[:, None, None]
    ) / NORM_STD[:, None, None]

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)


def create_tta_images(image: Image.Image) -> list[Image.Image]:
    """
    TTA equivalent to the configured validation views:

    1. Original
    2. Horizontal flip
    3. Vertical flip
    4. Rotate 90
    5. Rotate 270
    """
    image = image.convert("RGB")

    return [
        image,
        ImageOps.mirror(image),
        ImageOps.flip(image),
        image.rotate(90, expand=True),
        image.rotate(270, expand=True),
    ]


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    logits = logits.astype(np.float32)

    logits = logits - np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    exp_values = np.exp(logits)

    return exp_values / np.sum(
        exp_values,
        axis=-1,
        keepdims=True,
    )


def run_single_inference(image: Image.Image) -> np.ndarray:
    """
    Run one ONNX inference.

    Returns:
        probabilities with shape (num_classes,)
    """
    input_tensor = preprocess_image(image)

    outputs = SESSION.run(
        OUTPUT_NAMES,
        {
            INPUT_NAME: input_tensor,
        },
    )

    raw_output = np.asarray(outputs[0])

    # Expected shape is usually:
    # (1, 5)
    if raw_output.ndim == 2:
        logits = raw_output[0]
    elif raw_output.ndim == 1:
        logits = raw_output
    else:
        raise RuntimeError(
            f"Unexpected ONNX output shape: {raw_output.shape}"
        )

    if logits.shape[0] != len(class_names):
        raise RuntimeError(
            "Model output class count does not match "
            f"class_names.json: "
            f"{logits.shape[0]} vs {len(class_names)}"
        )

    return softmax(logits)


def predict(
    image: Image.Image,
    use_tta: bool = True,
) -> dict[str, Any]:
    """
    Run prediction with optional TTA.

    Args:
        image: PIL image.
        use_tta:
            True  -> 5-view TTA
            False -> single-image inference
    """
    if use_tta:
        images = create_tta_images(image)

        predictions = [
            run_single_inference(view)
            for view in images
        ]

        # Average probabilities across TTA views
        probabilities = np.mean(
            np.stack(predictions, axis=0),
            axis=0,
        )

    else:
        probabilities = run_single_inference(image)

    probabilities = probabilities.astype(float)

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    # Top 3
    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    top_3 = []

    for index in top_indices:
        class_name = class_names[int(index)]

        top_3.append(
            {
                "class": class_name,
                "label": class_labels.get(
                    class_name,
                    class_name,
                ),
                "confidence": round(
                    float(probabilities[index]),
                    6,
                ),
            }
        )

    return {
        "predicted_class": predicted_class,
        "predicted_label": class_labels.get(
            predicted_class,
            predicted_class,
        ),
        "confidence": round(confidence, 6),
        "top_3": top_3,
        "tta_used": use_tta,
        "tta_views": 5 if use_tta else 1,
    }


def model_info() -> dict[str, Any]:
    """
    Return useful model information.
    """
    return {
        "model": "EfficientNet-B3",
        "format": "ONNX",
        "input_size": IMG_SIZE,
        "classes": class_names,
        "num_classes": len(class_names),
        "normalization": {
            "mean": NORM_MEAN.tolist(),
            "std": NORM_STD.tolist(),
        },
        "providers": SESSION.get_providers(),
        "input_name": INPUT_NAME,
        "output_names": OUTPUT_NAMES,
    }