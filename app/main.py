"""Web app for the waste classifier.

Run from the repo root:
    uvicorn app.main:app --reload
Set WASTE_MODEL_PATH to use a model file other than models/waste_classifier.keras.
"""
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from waste_classifier import CLASS_INFO, load_classifier, load_image

APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = APP_DIR / "samples"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["classifier"] = load_classifier(os.environ.get("WASTE_MODEL_PATH"))
    yield


app = FastAPI(title="Waste Classifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/status")
def status():
    clf = state["classifier"]
    return {"backend": clf.backend, "description": clf.description, "model_name": clf.model.name,
            "metrics": clf.metrics, "classes": CLASS_INFO}


@app.get("/api/samples")
def samples():
    return json.loads((SAMPLES_DIR / "samples.json").read_text())


@app.post("/api/classify")
async def classify(file: UploadFile):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That image is over 10 MB. Try a smaller photo.")
    try:
        img = load_image(io.BytesIO(data))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "That file isn't an image we can read. Use a JPG, PNG or WebP photo.")
    prediction = await run_in_threadpool(state["classifier"].predict, img)
    return prediction.to_dict()
