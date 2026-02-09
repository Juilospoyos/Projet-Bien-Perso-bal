from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import CamembertForSequenceClassification, CamembertTokenizer

# -----------------------------
# App and CORS configuration
# -----------------------------
app = FastAPI(title="TrashTalk Detection API", version="1.0.0")

# Allow CORS for browser integration (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Model loading (once at startup)
# -----------------------------
MODEL_DIR = Path(__file__).resolve().parent / "model"

_tokenizer: Optional[CamembertTokenizer] = None
_model: Optional[CamembertForSequenceClassification] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        if not MODEL_DIR.exists():
            raise RuntimeError(f"Model directory not found: {MODEL_DIR}")
        _tokenizer = CamembertTokenizer.from_pretrained(str(MODEL_DIR))
        _model = CamembertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()
        _model.to(_device)


@app.on_event("startup")
def _startup_event():
    # Preload model at startup for lower latency on first request
    load_artifacts()


# -----------------------------
# Schemas
# -----------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à analyser")


class PredictResponse(BaseModel):
    is_trashtalk: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str  # "trashtalk" or "not_trashtalk"


# -----------------------------
# Core inference
# -----------------------------
@torch.inference_mode()
def predict_trash(text: str) -> PredictResponse:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Le champ 'text' est vide.")

    load_artifacts()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    outputs = _model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]

    # Assuming label id 1 == trashtalk, 0 == not_trashtalk (as in your previous code)
    trashtalk_prob = float(probs[1].item())
    is_trashtalk = trashtalk_prob >= 0.5

    return PredictResponse(
        is_trashtalk=is_trashtalk,
        confidence=trashtalk_prob if is_trashtalk else float(probs[0].item()),
        label="trashtalk" if is_trashtalk else "not_trashtalk",
    )


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    load_artifacts()
    return {
        "status": "ok",
        "device": str(_device),
        "model_dir": str(MODEL_DIR),
        "model_loaded": _model is not None,
        "tokenizer_loaded": _tokenizer is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    try:
        return predict_trash(payload.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")


# -----------------------------
# Local development entrypoint
# -----------------------------
if __name__ == "__main__":
    # Start with: python api_trash_ia.py
    import uvicorn

    uvicorn.run(
        "api_trash_ia:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
