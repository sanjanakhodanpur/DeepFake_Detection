"""
train_xception_lstm.py

End-to-end training script for the Xception + LSTM deepfake detector.

This script is intentionally long and heavily commented so that:
  - It is easy to understand for report / viva.
  - It can be handed to a client as explicit documentation of the pipeline.
  - It can be modified later (changing dataset, hyperparameters, etc).

Pipeline overview
=================
1. Read images from folders:
       data/train/real
       data/train/fake
       data/val/real
       data/val/fake

2. Build a tf.data.Dataset for train and validation:
       - load images
       - apply data augmentation (for robustness)
       - normalize pixel range to [-1, 1] as required by Xception

3. Convert each image into a short "sequence" for the LSTM:
       - we expand the shape (H, W, C) -> (TIMESTEPS, H, W, C)
       - the same frame is repeated TIMESTEPS times
       - this keeps the architecture (Xception+LSTM) but still works with images

4. Build the Xception + LSTM model using the function from deepfake_detector.py

5. Train the model with:
       - binary cross-entropy loss
       - accuracy metric
       - callbacks:
           * ModelCheckpoint   -> saves best model to models/xception_lstm.h5
           * EarlyStopping     -> stops training if val_loss does not improve
           * ReduceLROnPlateau -> reduce learning rate when stuck

6. Save training history and print a small summary of metrics.

After running this script successfully you will have:
       models/xception_lstm.h5

This file is then used by your Flask app (app.py) through deepfake_detector.py
for real-time predictions on uploaded images or videos.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
from typing import Tuple, Dict, Any

import tensorflow as tf

# We reuse architecture & constants from your existing file
from deepfake_detector import (
    build_xception_lstm_model,
    IMG_SIZE,
    TIMESTEPS,
    MODEL_WEIGHTS_PATH,
)

# ---------------------------------------------------------------------------
# 0. HIGH-LEVEL CONFIGURATION
# ---------------------------------------------------------------------------


class TrainConfig:
    """
    Configuration object for the training run.

    You can tweak values here and rerun the script without touching the rest
    of the code.
    """

    # Root folder where the dataset is stored.
    DATA_DIR = "data"

    # Train/validation subdirectories
    TRAIN_SUBDIR = "train"
    VAL_SUBDIR = "val"

    # Class names must match folder names inside train/ and val/
    CLASS_NAMES = ("real", "fake")

    # Model & training parameters
    BATCH_SIZE = 2
    EPOCHS = 10  # increase if you have time and GPU
    SHUFFLE_BUFFER = 1024

    # data augmentation parameters
    USE_DATA_AUGMENTATION = True

    # callbacks configuration
    EARLY_STOPPING_PATIENCE = 3
    LR_PATIENCE = 2
    LR_FACTOR = 0.5
    MIN_LR = 1e-6

    # file to save training history
    HISTORY_JSON_PATH = os.path.join("models", "xception_lstm_history.json")


CFG = TrainConfig()


# ---------------------------------------------------------------------------
# 1. DATASET UTILITIES
# ---------------------------------------------------------------------------

def verify_dataset_structure(base_dir: str) -> None:
    """
    Verify that the expected folders exist. If not, raise a helpful error.
    """
    train_real = os.path.join(base_dir, CFG.TRAIN_SUBDIR, "real")
    train_fake = os.path.join(base_dir, CFG.TRAIN_SUBDIR, "fake")
    val_real = os.path.join(base_dir, CFG.VAL_SUBDIR, "real")
    val_fake = os.path.join(base_dir, CFG.VAL_SUBDIR, "fake")

    missing = []
    for path in (train_real, train_fake, val_real, val_fake):
        if not os.path.isdir(path):
            missing.append(path)

    if missing:
        msg_lines = [
            "ERROR: The following required dataset directories are missing:"
        ]
        for m in missing:
            msg_lines.append(f"  - {m}")
        msg_lines.append("")
        msg_lines.append("Expected structure:")
        msg_lines.append("  data/")
        msg_lines.append("    train/")
        msg_lines.append("      real/")
        msg_lines.append("      fake/")
        msg_lines.append("    val/")
        msg_lines.append("      real/")
        msg_lines.append("      fake/")
        raise RuntimeError("\n".join(msg_lines))

    print("✅ Dataset structure looks correct.")


def build_image_dataset(
    base_dir: str, subset: str
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a directory.

    Args:
        base_dir: root data directory (e.g., "data").
        subset: either "train" or "val".

    Returns:
        A dataset of (image, label) where image shape is (H, W, 3)
        and label is 0.0 (real) or 1.0 (fake).
    """
    assert subset in (CFG.TRAIN_SUBDIR, CFG.VAL_SUBDIR), "subset must be 'train' or 'val'"

    subset_dir = os.path.join(base_dir, subset)
    print(f"\n📁 Building {subset} dataset from: {subset_dir}")

    ds = tf.keras.utils.image_dataset_from_directory(
        subset_dir,
        labels="inferred",            # folder names -> labels
        label_mode="binary",          # single scalar 0/1
        class_names=list(CFG.CLASS_NAMES),
        batch_size=CFG.BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=(subset == CFG.TRAIN_SUBDIR),
    )

    # Show class mapping (for explicit documentation)
    print("Class indices:", ds.class_names)

    return ds


def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Create a Keras Sequential of data augmentation layers.

    This will only be applied to the training dataset.
    """
    if not CFG.USE_DATA_AUGMENTATION:
        # Return a dummy "identity" layer
        return tf.keras.Sequential(name="no_augmentation")

    aug_layers = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(factor=0.1),
        ],
        name="data_augmentation",
    )
    return aug_layers


def preprocess_dataset(
    ds: tf.data.Dataset, augment: bool
) -> tf.data.Dataset:
    """
    Apply preprocessing steps to a dataset:

      - optional augmentation
      - normalization to [-1, 1]
      - caching & prefetching for performance
    """
    aug = build_augmentation_layer()

    def _preprocess(image, label):
        # image is (B, H, W, C) when called through map on a batched dataset
        if augment and CFG.USE_DATA_AUGMENTATION:
            image = aug(image)
        # Normalize from [0,255] -> [-1,1]
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1.0
        return image, label

    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def add_time_dimension(
    ds: tf.data.Dataset,
) -> tf.data.Dataset:
    """
    Convert images (B, H, W, C) to sequences (B, T, H, W, C)
    by repeating each image TIMESTEPS times.

    This allows us to use the same Xception+LSTM architecture for an image
    dataset as if each image was a tiny video clip.
    """

    def _to_sequence(images, labels):
        # images: (B, H, W, C)
        # expand to (B, 1, H, W, C)
        images = tf.expand_dims(images, axis=1)
        # tile along time dimension: (B, T, H, W, C)
        images = tf.tile(images, [1, TIMESTEPS, 1, 1, 1])
        return images, labels

    ds = ds.map(_to_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# 2. CALLBACKS
# ---------------------------------------------------------------------------

def build_callbacks() -> list:
    """
    Build a list of Keras callbacks:
      - ModelCheckpoint: saves best model to MODEL_WEIGHTS_PATH
      - EarlyStopping: stop if val_loss stops improving
      - ReduceLROnPlateau: reduce LR when stuck
    """
    callbacks = []

    # Make sure models folder exists
    os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)

    # 1) checkpoint
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_WEIGHTS_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    callbacks.append(ckpt_cb)

    # 2) early stopping
    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CFG.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(es_cb)

    # 3) reduce LR on plateau
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=CFG.LR_FACTOR,
        patience=CFG.LR_PATIENCE,
        min_lr=CFG.MIN_LR,
        verbose=1,
    )
    callbacks.append(lr_cb)

    return callbacks


# ---------------------------------------------------------------------------
# 3. TRAINING LOGIC
# ---------------------------------------------------------------------------

def train() -> Dict[str, Any]:
    """
    Main function to perform the end-to-end training.

    Returns:
        history_dict: a dictionary of training history (loss, accuracy, etc.)
    """
    print("\n=========================== TRAINING CONFIG ===========================")
    print(f"DATA_DIR:          {CFG.DATA_DIR}")
    print(f"IMG_SIZE:          {IMG_SIZE}")
    print(f"TIMESTEPS:         {TIMESTEPS}")
    print(f"BATCH_SIZE:        {CFG.BATCH_SIZE}")
    print(f"EPOCHS:            {CFG.EPOCHS}")
    print(f"USE_AUGMENTATION:  {CFG.USE_DATA_AUGMENTATION}")
    print(f"MODEL_WEIGHTS:     {MODEL_WEIGHTS_PATH}")
    print("=====================================================================\n")

    # 1) Verify dataset exists
    verify_dataset_structure(CFG.DATA_DIR)

    # 2) Build raw datasets from directories
    raw_train_ds = build_image_dataset(CFG.DATA_DIR, CFG.TRAIN_SUBDIR)
    raw_val_ds = build_image_dataset(CFG.DATA_DIR, CFG.VAL_SUBDIR)

    # 3) Preprocess (augment + normalize)
    train_ds = preprocess_dataset(raw_train_ds, augment=True)
    val_ds = preprocess_dataset(raw_val_ds, augment=False)

    # 4) Add time dimension for LSTM
    train_seq = add_time_dimension(train_ds)
    val_seq = add_time_dimension(val_ds)

    # 5) Build model
    print("\n🧠 Building Xception + LSTM model ...")
    model = build_xception_lstm_model()
    model.summary()

    # 6) Train
    callbacks = build_callbacks()

    print("\n🚀 Starting training ...")
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=CFG.EPOCHS,
        callbacks=callbacks,
    )

    print("\n✅ Training complete.")
    print(f"Best model saved to: {MODEL_WEIGHTS_PATH}")

    # 7) Convert history to a plain Python dict for JSON saving
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}

    return history_dict


# ---------------------------------------------------------------------------
# 4. UTILITY: SAVE HISTORY TO JSON
# ---------------------------------------------------------------------------

def save_history(history: Dict[str, Any], path: str) -> None:
    """
    Save training history (loss, accuracy, etc.) as JSON for later analysis
    or plotting.

    Args:
        history: dictionary obtained from model.fit().history
        path: output JSON file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n📁 Training history saved to: {path}")


def print_history_summary(history: Dict[str, Any]) -> None:
    """
    Print a small summary of best validation metrics for quick inspection.
    """
    if not history:
        print("No history to summarize.")
        return

    def best_metric(name: str) -> Tuple[float, int]:
        values = history.get(name, None)
        if not values:
            return float("nan"), -1
        best_val = min(values) if "loss" in name else max(values)
        best_epoch = int(values.index(best_val)) + 1
        return best_val, best_epoch

    val_loss, ep_loss = best_metric("val_loss")
    val_acc, ep_acc = best_metric("val_accuracy")

    print("\n======================== HISTORY SUMMARY ========================")
    if not (val_loss != val_loss):  # NaN check
        print(f"Best val_loss:     {val_loss:.4f} (epoch {ep_loss})")
    if not (val_acc != val_acc):
        print(f"Best val_accuracy: {val_acc:.4f} (epoch {ep_acc})")
    print("================================================================\n")


# ---------------------------------------------------------------------------
# 5. MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """
    Entry point when you run:

        python train_xception_lstm.py

    This function orchestrates the entire training process.
    """
    # Limit TF logging noise (optional)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    # Actually train
    history_dict = train()

    # Save history
    save_history(history_dict, CFG.HISTORY_JSON_PATH)

    # Print a simple summary
    print_history_summary(history_dict)

    print("🎉 All done! You can now use models/xception_lstm.h5 in your Flask app.")


if __name__ == "__main__":
    main()

    
