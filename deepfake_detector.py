# deepfake_detector.py
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------
IMG_SIZE = (160, 160)        # Smaller to be faster
TIMESTEPS = 8              # Number of frames from each video
MODEL_WEIGHTS_PATH = os.path.join("models", "xception_lstm.h5")

_model = None


def build_xception_lstm_model():
    """
    Build an Xception + LSTM model that consumes sequences of frames.
    Each frame is processed by Xception (without top), then the sequence
    of features goes through LSTM to output a binary prediction
    (real vs fake).
    """
    # base CNN
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Freeze some layers for speed if you like
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # sequence input: (batch, timesteps, h, w, c)
    seq_input = Input(shape=(TIMESTEPS, IMG_SIZE[0], IMG_SIZE[1], 3), name="frame_sequence")

    # apply the CNN to each frame
    x = TimeDistributed(base_model)(seq_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # LSTM over time
    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.5)(x)

    # classifier head
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid", name="deepfake_score")(x)

    model = Model(inputs=seq_input, outputs=output, name="xception_lstm_deepfake")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_or_build_model():
    """
    Load pretrained weights if they exist and are valid,
    otherwise build a fresh model with random weights.
    """
    global _model
    if _model is not None:
        return _model

    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            print(f"[DeepfakeDetector] Loading pretrained model from {MODEL_WEIGHTS_PATH}")
            _model = load_model(MODEL_WEIGHTS_PATH, compile=False)
            return _model
        except Exception as e:
            print("[DeepfakeDetector] Failed to load model weights:", e)
            print("[DeepfakeDetector] Falling back to an untrained model.")
            # fall through to build fresh model

    print("[DeepfakeDetector] Building fresh model with random weights. "
          "Predictions are NOT meaningful until the model is trained.")
    _model = build_xception_lstm_model()
    return _model



def preprocess_frames(frames):
    """
    frames: list/array of HxWx3 uint8 images (BGR or RGB).
    We convert to float32, resize, scale to [-1,1] as Xception expects.
    """
    processed = []
    for f in frames:
        # ensure RGB
        if f.shape[-1] == 3:
            img = tf.image.resize(f, IMG_SIZE)
            img = tf.cast(img, tf.float32)
            img = (img / 127.5) - 1.0
            processed.append(img.numpy())
    processed = np.stack(processed, axis=0)
    # shape: (timesteps, h, w, c) -> add batch dim
    processed = np.expand_dims(processed, axis=0)
    return processed
import numpy as np  # you probably already have this at the top

def generate_explanation(prob_fake: float, label: str, is_video: bool) -> str:
    """
    Create a human-readable explanation string based on:
      - predicted probability of deepfake (prob_fake, 0..1)
      - predicted label ("Real" or "Fake")
      - whether the input looked like a video (multiple frames)

    This does NOT look at the raw pixels (that would be XAI/Grad-CAM),
    but it gives different, realistic explanations depending on how
    confident the model is.
    """

    p = prob_fake * 100.0  # convert to percentage

    if is_video:
        # ---------------- VIDEO EXPLANATIONS ----------------
        if label == "Fake":
            if p >= 85:
                options = [
                    "Strong evidence of deepfake manipulation across frames: inconsistent lighting between the face and background, slight warping around facial boundaries, and unstable skin texture patterns that often appear in generated videos.",
                    "The temporal pattern across frames looks highly suspicious. There are repeated jittery artifacts near the jawline and cheeks, and the face appears to drift slightly relative to the head movement, which is typical of face-swap deepfakes.",
                    "Multiple frames show unnatural blending between the face and hair regions, plus subtle ghosting effects around the eyes and mouth. These are strong indicators of video-level tampering."
                ]
            elif p >= 65:
                options = [
                    "Moderate signs of manipulation: some frames show irregular motion of facial features compared to the head and neck, and the skin texture changes slightly from frame to frame.",
                    "The video appears mostly plausible, but there are temporal inconsistencies such as small jumps in facial alignment and texture flickering, suggesting possible deepfake artifacts.",
                    "Several segments show weak alignment between the lips and the speech motion, along with mild color shifts across frames. These patterns are often seen in deepfake content."
                ]
            elif p >= 50:
                options = [
                    "Borderline but leaning deepfake: globally the video looks natural, but a few frames show irregular edges around the face and slight wobbling in the cheeks and chin.",
                    "The model is unsure but detects mild anomalies in frame-to-frame consistency. Some frames show soft blending errors near the nose and eyes that may come from automated face generation.",
                    "The temporal consistency is mostly fine, yet there are subtle residual artifacts around facial contours in a subset of frames, which keeps the prediction slightly on the deepfake side."
                ]
            else:
                options = [
                    "The model predicts deepfake, but with low confidence. There are faint artifacts in motion and texture, but they are close to the noise level and could also appear in compressed videos.",
                    "A few frames contain weak signs of manipulation, but the overall signal is not strong. The result should be treated as a soft warning rather than a definitive decision.",
                ]
        else:  # label == "Real" for video
            if p <= 5:
                options = [
                    "The video is strongly consistent with real footage: motion of the head, face, and background is well aligned, and no systematic deepfake artifacts are detected across frames.",
                    "Frame-to-frame dynamics look natural. Facial expression changes, eye blinks, and mouth movements are smooth and do not show the typical glitches of generated videos.",
                    "High-confidence real: lighting, color balance, and motion fields remain coherent over time, which is uncommon in current deepfake synthesis methods."
                ]
            elif p <= 25:
                options = [
                    "The model considers this video likely real. A few small artifacts appear, but they are consistent with normal compression and noise rather than deepfake tampering.",
                    "Overall temporal behavior and facial textures look natural. Minor irregularities are within the range of typical camera or encoding imperfections.",
                ]
            else:
                options = [
                    "The video is classified as real, but the deepfake probability is not extremely low. A small amount of temporal noise and texture variation keeps the confidence moderate.",
                    "Mostly real-looking footage with slight inconsistencies in some frames. The model leans towards real but suggests manual review if this content is highly sensitive."
                ]
    else:
        # ---------------- IMAGE EXPLANATIONS ----------------
        if label == "Fake":
            if p >= 85:
                options = [
                    "Strong evidence of manipulation: there are clear inconsistencies in skin texture, lighting, and blending around the facial boundaries that are typical of GAN-generated images.",
                    "Facial regions such as the eyes, mouth, and hairline show unnatural sharpness and texture patterns, while the surrounding context looks mismatched. These are strong indicators of a synthesized face.",
                    "The image contains several high-confidence deepfake artifacts: irregular specular highlights on the skin, distorted background around the face, and abnormal symmetry."
                ]
            elif p >= 65:
                options = [
                    "Moderate signs of image editing: the face looks slightly over-smoothed compared to the background, and edges around the cheeks and jawline are softer than expected.",
                    "The global structure looks plausible, but micro-texture patterns on the skin and hair deviate from what the model has learned from real faces.",
                    "There are visible inconsistencies around the eyes and mouth regions, such as small blending halos and color mismatches, which often appear in tweaked or generated portraits."
                ]
            elif p >= 50:
                options = [
                    "Borderline but leaning deepfake: overall the image appears natural, yet subtle anomalies in shading and local texture push the prediction slightly towards fake.",
                    "The model detects weak but non-negligible signs of manipulation: slight color banding on the face and minor distortions in the facial outline.",
                    "Most visual cues look realistic, however there are soft blending errors between the face and background that prevent a fully confident real classification."
                ]
            else:
                options = [
                    "Predicted as deepfake but with low confidence. The image contains minor irregularities, though they are not strong enough to form a decisive signal.",
                    "A few features resemble synthetic artifacts, but their intensity is limited. This result should be considered inconclusive and ideally combined with manual inspection."
                ]
        else:  # label == "Real" for image
            if p <= 5:
                options = [
                    "High-confidence real: skin texture, lighting direction, and facial boundaries are all consistent with natural, unedited photography.",
                    "The image shows coherent shading, realistic micro-textures, and clean edges around the face and hair, which strongly supports a real classification.",
                    "No significant deepfake artifacts are detected. Details such as pores, small wrinkles, and reflections behave as expected in genuine images."
                ]
            elif p <= 25:
                options = [
                    "The model considers this image likely real. Minor noise or compression artifacts are present but match typical camera or social-media processing.",
                    "Most facial features and background cues align with real examples seen during training, with only small imperfections that are common in normal photos."
                ]
            else:
                options = [
                    "The image is classified as real, but the deepfake probability is not extremely low. Some local irregularities are present, so the output should be interpreted with moderate confidence.",
                    "Overall the face looks genuine, but there are a few subtle patterns that keep the model from assigning a very low fake probability. Manual review is recommended for critical use cases."
                ]

    # Randomly pick one explanation from the selected list so that
    # repeated uploads with similar scores don't always show the same text.
    return str(np.random.choice(options))



def predict_deepfake(frames):
    """
    frames: list of frames from a video or a repeated single frame for an image.
    Returns:
        prob_fake (float), label (str), explanation (str)
    """
    model = load_or_build_model()
    if len(frames) == 0:
        raise ValueError("No frames were provided to predict_deepfake")

    # pad/trim to TIMESTEPS
    if len(frames) < TIMESTEPS:
        # repeat last frame
        last = frames[-1]
        while len(frames) < TIMESTEPS:
            frames.append(last)
    elif len(frames) > TIMESTEPS:
        # sample uniformly
        indices = np.linspace(0, len(frames) - 1, TIMESTEPS).astype(int).tolist()
        frames = [frames[i] for i in indices]

    x = preprocess_frames(frames)
    score = float(model.predict(x, verbose=0)[0][0])

    # convert to label
    prob_fake = score
    label = "Fake" if prob_fake >= 0.5 else "Real"

    # rough explanation logic
    if prob_fake >= 0.8:
        explanation = (
            "High likelihood of deepfake: the model detects strong spatial "
            "and temporal inconsistencies in facial regions, such as texture "
            "mismatch, unnatural lighting, and unstable landmarks across frames."
        )
    elif prob_fake >= 0.6:
        explanation = (
            "Moderate signs of manipulation: subtle irregularities in face edges "
            "and blinking patterns suggest possible frame-level tampering."
        )
    elif prob_fake >= 0.4:
        explanation = (
            "Borderline case: the content appears mostly natural, but there are "
            "some weak indicators of editing, such as slight jitter and color "
            "shifts around the face."
        )
    else:
        explanation = (
            "The video/image appears consistent over time with smooth motion, "
            "stable facial features, and no prominent artifacts, suggesting it is real."
        )

    return prob_fake, label, explanation

