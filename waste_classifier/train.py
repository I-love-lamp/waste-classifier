"""Train the waste classifier.

Expects the Kaggle "Waste Classification data" layout:
    DATASET/TRAIN/O/*.jpg   DATASET/TRAIN/R/*.jpg
    DATASET/TEST/O/*.jpg    DATASET/TEST/R/*.jpg

Without --data, the dataset is downloaded from Kaggle (and cached) via kagglehub.

Usage:
    python -m waste_classifier.train
    python -m waste_classifier.train --data path/to/DATASET
    python -m waste_classifier.train --arch mobilenet --epochs 5
    python -m waste_classifier.train --arch mobilenet --epochs 5 --fine-tune-layers 30 --fine-tune-epochs 3
"""
import argparse
import json
from pathlib import Path

import keras

from .dataset import download_dataset
from .evaluation import metrics_path, score_model
from .model import IMAGE_SIZE, build_cnn, build_mobilenet, unfreeze_top
from .predictor import DEFAULT_MODEL_PATH


def load_split(path: Path, batch_size: int, shuffle: bool, **kwargs):
    return keras.utils.image_dataset_from_directory(
        path, image_size=IMAGE_SIZE, batch_size=batch_size,
        label_mode="categorical", shuffle=shuffle, seed=42, **kwargs,
    )


def load_data(data: Path | None = None, batch_size: int = 64, val_split: float = 0.1):
    """Return (train, val, test) datasets. val is a held-out slice of TRAIN; test is TEST."""
    data = Path(data) if data else download_dataset()
    train_ds = load_split(data / "TRAIN", batch_size, shuffle=True, validation_split=val_split, subset="training")
    val_ds = load_split(data / "TRAIN", batch_size, shuffle=True, validation_split=val_split, subset="validation")
    test_ds = load_split(data / "TEST", batch_size, shuffle=False)
    return train_ds, val_ds, test_ds


def fit(model, train_ds, val_ds, epochs: int, patience: int = 5,
        fine_tune_layers: int = 0, fine_tune_epochs: int = 0, fine_tune_lr: float = 1e-5, verbose: int = 1):
    """Train the head, then optionally fine-tune the top of the backbone.

    Returns the combined per-epoch history (a dict of lists) across both phases.
    """
    def run(n_epochs):
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        return model.fit(train_ds.prefetch(2), validation_data=val_ds.prefetch(2),
                         epochs=n_epochs, callbacks=[stop], verbose=verbose).history

    history = run(epochs)
    history["phase"] = ["head"] * len(history["loss"])
    if fine_tune_layers and fine_tune_epochs:
        unfreeze_top(model, fine_tune_layers, fine_tune_lr)
        tuned = run(fine_tune_epochs)
        for key, values in tuned.items():
            history[key] += values
        history["phase"] += ["fine-tune"] * len(tuned["loss"])
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="dataset root containing TRAIN/ and TEST/ (default: download from Kaggle)")
    parser.add_argument("--arch", choices=["cnn", "mobilenet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="fraction of TRAIN held out for early stopping; TEST is only used for the final score")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, help="default: 0.5 for cnn, 0.2 for mobilenet")
    mobilenet = parser.add_argument_group("mobilenet only")
    mobilenet.add_argument("--hidden-units", type=int, default=0, help="extra dense layer in the head (0 = none)")
    mobilenet.add_argument("--augment", action="store_true", help="random flip/rotate/zoom during training")
    mobilenet.add_argument("--fine-tune-layers", type=int, default=0, help="backbone layers to unfreeze after head training")
    mobilenet.add_argument("--fine-tune-epochs", type=int, default=0)
    mobilenet.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    if args.arch == "cnn" and (args.hidden_units or args.augment or args.fine_tune_layers):
        parser.error("--hidden-units, --augment and --fine-tune-* only apply to --arch mobilenet")

    train_ds, val_ds, test_ds = load_data(args.data, args.batch_size, args.val_split)
    class_names = train_ds.class_names

    if args.arch == "cnn":
        model = build_cnn(len(class_names), dropout=args.dropout if args.dropout is not None else 0.5,
                          learning_rate=args.learning_rate)
    else:
        model = build_mobilenet(len(class_names), dropout=args.dropout if args.dropout is not None else 0.2,
                                hidden_units=args.hidden_units, learning_rate=args.learning_rate, augment=args.augment)
    model.summary()
    fit(model, train_ds, val_ds, args.epochs, args.patience,
        args.fine_tune_layers, args.fine_tune_epochs, args.fine_tune_lr)

    scores = score_model(model, test_ds, class_names)
    print("test " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items() if isinstance(v, float)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    args.out.with_suffix(".labels.json").write_text(json.dumps(class_names))
    metrics_path(args.out).write_text(json.dumps(scores, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
