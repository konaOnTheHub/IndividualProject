import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from api.labels import CLASS_NAMES, NUM_CLASSES
from api.model import load_waste_classifier, pick_device, resolve_checkpoint_path
from api.preprocess import build_eval_transform

MAX_UPLOAD_BYTES = int(os.environ.get("WASTE_MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/jpg"}
)


class ClassifyResponse(BaseModel):
    """
    Stable contract for mobile clients.
    """

    category: str = Field(..., description="Predicted waste class name.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_index: int = Field(..., ge=0, lt=NUM_CLASSES)
    probabilities: dict[str, float] = Field(
        ..., description="Softmax probability per class name."
    )


class HealthResponse(BaseModel):
    status: str
    device: str
    model_path: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = resolve_checkpoint_path()
    model, device = load_waste_classifier(ckpt_path=ckpt, device=pick_device())
    app.state.model = model
    app.state.device = device
    app.state.ckpt_path = str(ckpt)
    app.state.transform = build_eval_transform()
    yield


app = FastAPI(
    title="Waste classification API",
    description="EfficientNetV2-S (384px) waste image classifier.",
    lifespan=lifespan,
)

_cors = os.environ.get("CORS_ORIGINS", "").strip()
if _cors:
    origins = ["*"] if _cors == "*" else [o.strip() for o in _cors.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = hasattr(app.state, "model") and app.state.model is not None
    return HealthResponse(
        status="ok",
        device=str(app.state.device) if loaded else pick_device().type,
        model_path=getattr(app.state, "ckpt_path", str(resolve_checkpoint_path())),
        model_loaded=loaded,
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a waste image",
    description=(
        "Returns category and per-class probabilities. "
    ),
)
async def classify(file: UploadFile = File(...)):
    # check if the file is an image
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            415,
            detail=f"Unsupported media type {file.content_type!r}; "
            f"allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    # check if the file is too large
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, detail=f"File too large; max {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"
        )

    try:
        img = Image.open(io.BytesIO(data))
        # Match how users see the photo: camera JPEGs often rely on EXIF Orientation while
        # some clients (e.g. mobile pickers) pre-rotate and strip EXIF. Transposing here
        # makes raw uploads and app uploads classify the same scene consistently.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except OSError as e:
        raise HTTPException(400, detail=f"Invalid image: {e}") from e
    #Convert image to tensor
    tensor = app.state.transform(img).unsqueeze(0).to(app.state.device)
    #Load model
    model = app.state.model
    #Make prediction
    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
    #Get the predicted class and confidence
    conf, idx = torch.max(probs, dim=0)
    idx_int = int(idx.item())
    prob_list = probs.cpu().tolist()
    probabilities = {CLASS_NAMES[i]: float(prob_list[i]) for i in range(NUM_CLASSES)}
    #Return the prediction
    return ClassifyResponse(
        category=CLASS_NAMES[idx_int],
        confidence=float(conf.item()),
        class_index=idx_int,
        probabilities=probabilities,
    )
