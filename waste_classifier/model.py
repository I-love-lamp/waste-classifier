"""Model architectures for the waste classifier.

Both models take raw RGB images (0-255) at IMAGE_SIZE and do their own rescaling,
so training and inference cannot drift apart on preprocessing.
"""
import keras
from keras import layers

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
BACKBONE_NAME = "backbone"


def _compile(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _augmentation() -> keras.Sequential:
    """Random flips, small rotations and zooms. Active only during training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augment")


def build_cnn(num_classes: int = 2, dropout: float = 0.5, learning_rate: float = 1e-3) -> keras.Model:
    """The original notebook CNN: two conv blocks and a dense head.

    Changes from the notebook: rescaling lives inside the model, and the head uses
    softmax + categorical cross-entropy (the classes are mutually exclusive).
    """
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ], name="waste_cnn")
    return _compile(model, learning_rate)


def classifier_head(x, num_classes: int, dropout: float, hidden_units: int):
    """Dense head on top of pooled backbone features. Shared by the full model and
    the cached-feature models used for fast tuning, so tuned settings carry over."""
    x = layers.Dropout(dropout)(x)
    if hidden_units:
        x = layers.Dense(hidden_units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    return layers.Dense(num_classes, activation="softmax")(x)


def _backbone() -> keras.Model:
    base = keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False,
                                          weights="imagenet", name=BACKBONE_NAME)
    base.trainable = False
    return base


def build_mobilenet(num_classes: int = 2, dropout: float = 0.2, hidden_units: int = 0,
                    learning_rate: float = 1e-3, augment: bool = False) -> keras.Model:
    """Transfer-learning model: frozen ImageNet MobileNetV2 with a small head.

    Trains in a fraction of the time of the CNN and usually generalises better.
    Call unfreeze_top() afterwards to fine-tune the top of the backbone.
    """
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = _augmentation()(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    # training=False keeps BatchNorm statistics frozen, including while fine-tuning.
    x = _backbone()(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = classifier_head(x, num_classes, dropout, hidden_units)
    return _compile(keras.Model(inputs, outputs, name="waste_mobilenet"), learning_rate)


def build_feature_extractor() -> keras.Model:
    """Raw image -> pooled MobileNetV2 features (1280-d). Matches build_mobilenet's preprocessing."""
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = _backbone()(x, training=False)
    return keras.Model(inputs, layers.GlobalAveragePooling2D()(x), name="mobilenet_features")


def build_head(feature_dim: int, num_classes: int = 2, dropout: float = 0.2, hidden_units: int = 0,
               learning_rate: float = 1e-3) -> keras.Model:
    """The build_mobilenet head on its own, for training on cached features."""
    inputs = keras.Input(shape=(feature_dim,))
    outputs = classifier_head(inputs, num_classes, dropout, hidden_units)
    return _compile(keras.Model(inputs, outputs, name="head"), learning_rate)


def unfreeze_top(model: keras.Model, n_layers: int, learning_rate: float = 1e-5) -> keras.Model:
    """Make the top n_layers of the MobileNetV2 backbone trainable and recompile at a low learning rate."""
    base = model.get_layer(BACKBONE_NAME)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    return _compile(model, learning_rate)


ARCHITECTURES = {"cnn": build_cnn, "mobilenet": build_mobilenet}
