"""Inference: turn an image into an organic/recyclable prediction.

Two backends share one interface:
- TrainedClassifier: a model produced by `python -m waste_classifier.train`.
- ImageNetDemoClassifier: used when no trained model exists. It runs a stock ImageNet
  MobileNetV2 and sums the probability of food/plant classes as "organic". It is a
  stand-in so the app works out of the box, not a substitute for training.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .labels import CLASS_INFO, CLASSES
from .model import IMAGE_SIZE

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "waste_classifier.keras"


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    backend: str
    # ImageNet labels the demo backend based its decision on; empty for trained models.
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        info = CLASS_INFO[self.label]
        return {
            "label": self.label,
            "name": info["name"],
            "bin": info["bin"],
            "advice": info["advice"],
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "backend": self.backend,
            "evidence": self.evidence,
        }


def load_image(source) -> Image.Image:
    """Open a path or file-like object as an upright RGB image."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def to_batch(img: Image.Image) -> np.ndarray:
    """Resize to the model input size and return a (1, H, W, 3) float array in 0-255."""
    resized = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[np.newaxis]


def _prediction(probs: dict[str, float], backend: str, evidence=None) -> Prediction:
    label = max(probs, key=probs.get)
    return Prediction(label, probs[label], probs, backend, evidence or [])


class TrainedClassifier:
    backend = "trained"

    def __init__(self, model_path: Path):
        import keras

        self.model_path = Path(model_path)
        self.model = keras.models.load_model(self.model_path)
        labels_path = self.model_path.with_suffix(".labels.json")
        self.classes = json.loads(labels_path.read_text()) if labels_path.exists() else CLASSES
        metrics_path = self.model_path.with_suffix(".metrics.json")
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        self.description = f"Trained model ({self.model.name}) from {self.model_path.name}"

    def predict(self, img: Image.Image) -> Prediction:
        out = self.model.predict(to_batch(img), verbose=0)[0]
        probs = {c: float(p) for c, p in zip(self.classes, out)}
        return _prediction(probs, self.backend)


# ImageNet class indices treated as organic: crabs and lobsters (118-124, shells are
# food waste), foods, fruit and veg (924-965), and plants, fungi and seeds (984-998).
# Everything else counts as recyclable, mirroring the training data where R means
# "anything not organic".
_ORGANIC_IMAGENET = set(range(118, 125)) | set(range(924, 966)) | set(range(984, 999))
_TOP_K = 5


class ImageNetDemoClassifier:
    backend = "demo"
    description = "Demo mode: ImageNet MobileNetV2 with a food/plant heuristic"
    metrics = None

    def __init__(self):
        import keras

        self._keras = keras
        self.model = keras.applications.MobileNetV2(weights="imagenet")

    def predict(self, img: Image.Image) -> Prediction:
        apps = self._keras.applications.mobilenet_v2
        out = self.model.predict(apps.preprocess_input(to_batch(img)), verbose=0)
        # Vote with the top guesses only: summing all 1000 classes lets the long tail
        # of non-food classes outweigh a confident "banana".
        top_idx = np.argsort(out[0])[::-1][:_TOP_K]
        names = apps.decode_predictions(out, top=_TOP_K)[0]
        evidence = [
            {"label": name.replace("_", " "), "score": float(score), "counts_as": "O" if i in _ORGANIC_IMAGENET else "R"}
            for (_, name, score), i in zip(names, top_idx)
        ]
        total = sum(e["score"] for e in evidence)
        organic = sum(e["score"] for e in evidence if e["counts_as"] == "O") / total
        probs = {"O": organic, "R": 1.0 - organic}
        return _prediction(probs, self.backend, evidence)


def load_classifier(model_path: Path | str | None = None):
    """Load the trained model if present, otherwise fall back to the demo backend."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        return TrainedClassifier(path)
    return ImageNetDemoClassifier()
