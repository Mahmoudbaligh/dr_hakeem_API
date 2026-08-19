from __future__ import annotations

import io
import time

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .inference import model_info, predict


app = FastAPI(
    title="Dr. Hakeem Skin Disease AI API",
    description=(
        "EfficientNet-B3 ONNX inference API "
        "for skin lesion classification."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "service": "Dr. Hakeem Skin Disease AI API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "predict_endpoint": "/predict",
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ============================================================
# Model Information
# ============================================================

@app.get("/model-info")
def get_model_info():
    return {
        "success": True,
        **model_info(),
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    tta: bool = Query(
        True,
        description=(
            "Use 5-view test-time augmentation."
        ),
    ),
):
    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type.",
        )

    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }

    if file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image type. "
                "Use JPEG, PNG, or WEBP."
            ),
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Force actual image decoding
        image.load()

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:
        result = predict(
            image=image,
            use_tta=tta,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    elapsed = time.perf_counter() - start_time

    return {
        "success": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "inference_time_ms": round(
            elapsed * 1000,
            2,
        ),
        **result,
    }