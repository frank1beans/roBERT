
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..extraction import ExtractionRouter
from ..inference.calibration import TemperatureCalibrator
from ..inference.predict_category import _load_id2label, load_classifier
from ..registry import load_pack

APP_VERSION = "0.1.0"

# ---- Env config ----
_DEFAULT_PACK_PATH = Path(__file__).resolve().parents[2] / "pack" / "current"
ENV_PACK = os.getenv(
    "ROBIMB_PACK",
    os.getenv("ROBIMB_PACK_CURRENT", str(_DEFAULT_PACK_PATH)),
)
ENV_MODEL = os.getenv("ROBIMB_MODEL", "runs/label")
ENV_LABEL_INDEX = os.getenv("ROBIMB_LABEL_INDEX", "data/wbs/label_maps.json")
ENV_CALIBRATOR = os.getenv("ROBIMB_CALIBRATOR")  # opzionale

# ---- Singletons with lock ----
_pack_lock = threading.Lock()
_model_lock = threading.Lock()
_state = {"pack": None, "model": None, "tokenizer": None, "id2label": None, "calibrator": None}

def _load_pack_once(path: str):
    with _pack_lock:
        if _state["pack"] is None or getattr(_state["pack"], "_path", None) != path:
            pack = load_pack(path)
            setattr(pack, "_path", path)
            _state["pack"] = pack
    return _state["pack"]

def _load_model_once(model_path: str, label_index_path: str, calibrator_path: Optional[str] = None):
    with _model_lock:
        need = (
            _state["model"] is None or
            getattr(_state["model"], "_path", None) != model_path or
            _state["id2label"] is None or
            getattr(_state["id2label"], "_path", None) != label_index_path or
            (calibrator_path and ( _state["calibrator"] is None or getattr(_state["calibrator"], "_path", None) != calibrator_path ))
        )
        if need:
            tok, mdl = load_classifier(model_path)
            id2label = _load_id2label(label_index_path)
            calibrator = None
            if calibrator_path and os.path.exists(calibrator_path):
                with open(calibrator_path,"r",encoding="utf-8") as f: sd=json.load(f)
                calibrator = TemperatureCalibrator.from_state_dict(sd)
                setattr(calibrator, "_path", calibrator_path)
            _state["tokenizer"] = tok
            _state["model"] = mdl; setattr(_state["model"], "_path", model_path)
            _state["id2label"] = id2label; setattr(_state["id2label"], "_path", label_index_path)
            _state["calibrator"] = calibrator
    return _state["model"], _state["tokenizer"], _state["id2label"], _state["calibrator"]

# ---- FastAPI app ----
app = FastAPI(title="roBIMB API", version=APP_VERSION)

class Context(BaseModel):
    exposure_env: Optional[str] = None
    climate_zone_UNI10349: Optional[str] = None

class PredictIn(BaseModel):
    text: str = Field(..., description="Descrizione grezza")
    topk: int = Field(5, ge=1, le=20)
    context: Optional[Context] = None
    # override opzionali
    pack_path: Optional[str] = None
    model_path: Optional[str] = None
    label_index_path: Optional[str] = None
    calibrator_path: Optional[str] = None

class PredictOut(BaseModel):
    category: Dict[str, Any]
    topk: List[Dict[str, Any]]
    properties: Dict[str, Any]
    issues: List[Dict[str, Any]]
    description: str
    input_text: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "env": {
            "pack": ENV_PACK, "model": ENV_MODEL,
            "label_index": ENV_LABEL_INDEX, "calibrator": ENV_CALIBRATOR
        }
    }

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    pack_path = payload.pack_path or ENV_PACK
    model_path = payload.model_path or ENV_MODEL
    label_index_path = payload.label_index_path or ENV_LABEL_INDEX
    calibrator_path = payload.calibrator_path or ENV_CALIBRATOR

    pack = _load_pack_once(pack_path)
    model, tokenizer, id2label, calibrator = _load_model_once(model_path, label_index_path, calibrator_path)

    # Reuse the same code path of CLI pipeline but pass preloaded objects via env vars
    from ..inference.predict_category import predict_topk as _predict_topk
    from ..templates.render import render

    top, topk_list, probs, logits = _predict_topk(payload.text, model, tokenizer, id2label, topk=payload.topk, calibrator=calibrator)
    ctx = payload.context.dict() if payload.context else {}
    router = ExtractionRouter(pack)
    router_output = router.extract(payload.text, categories=top["label"], context=ctx)
    props = router_output.postprocess.values
    issues = router_output.postprocess.issues or []
    descr = render(top["label"], props, pack.templates)

    return PredictOut(
        input_text=payload.text,
        category=top,
        topk=topk_list,
        properties=props,
        issues=issues,
        description=descr
    )
