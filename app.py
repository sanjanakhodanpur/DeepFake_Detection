"""
app.py

Main Flask application for the DeepGuard / Deepfake Detection UI.

This file:
  - Configures Flask and the upload folder
  - Serves the UI pages (home, about, contact)
  - Provides an /analyze endpoint for image/video deepfake detection
  - Provides a /chat endpoint for a simple assistant chatbot
"""

import os
import uuid
from typing import Tuple
from preprocessing.preprocess import preprocess_image

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    url_for,
)

from deepfake_detector import predict_deepfake
from video_utils import (
    is_image_file,
    is_video_file,
    load_image_frames,
    load_video_frames,
)

# -------------------------------------------------------------------
# 1. Flask setup and configuration
# -------------------------------------------------------------------

# create the Flask app
app = Flask(__name__)

# base folder of the project (where this app.py lives)
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# where uploaded media will be stored
UPLOAD_FOLDER: str = os.path.join(BASE_DIR, "static", "uploads")

# make sure the uploads folder exists (create if not)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask configuration
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# limit uploads to 500MB just to be safe
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------

def save_uploaded_file(file_storage) -> Tuple[str, str]:
    """
    Save a Werkzeug FileStorage object to the uploads folder.

    Returns:
        (full_path, filename)
        full_path: absolute path on disk
        filename: file name only (used for building URLs)
    """
    # get extension, convert to lowercase (e.g., .JPG -> .jpg)
    _, ext = os.path.splitext(file_storage.filename)
    ext = ext.lower()

    # random name so files don't collide
    random_name = f"{uuid.uuid4().hex}{ext}"
    full_path = os.path.join(UPLOAD_FOLDER, random_name)

    # actually save to disk
    file_storage.save(full_path)

    return full_path, random_name


# -------------------------------------------------------------------
# 3. Page routes (HTML pages)
# -------------------------------------------------------------------

@app.route("/")
def index():
    """
    Home page.
    Contains the main dashboard: upload area + animated screen + chatbot.
    """
    return render_template("index.html")


@app.route("/about")
def about():
    """
    About page: explains the project and model idea.
    """
    return render_template("about.html")


@app.route("/contact")
def contact():
    """
    Contact page: dummy form and project details.
    """
    return render_template("contact.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """
    Static route to serve uploaded media back to the browser.
    Example: /uploads/some-file.mp4
    """
    return send_from_directory(UPLOAD_FOLDER, filename)


# -------------------------------------------------------------------
# 4. Deepfake analysis API
# -------------------------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze

    Expects a file under the form field name "media".
    Handles both images and videos and returns JSON.
    """
    try:
        # ---------- Basic validation ----------
        if "media" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["media"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # ---------- Save file ----------
        saved_path, filename = save_uploaded_file(file)
        media_url = url_for("uploaded_file", filename=filename)

        # ---------- Decide image vs video ----------
        if is_image_file(filename):
            frames = load_image_frames(saved_path)
            analysis_type = "image"
        elif is_video_file(filename):
            # you can tweak max_frames/every_n if you want faster/slower analysis
            frames = load_video_frames(saved_path, max_frames=None, every_n=3)
            analysis_type = "video"
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # ---------- Model prediction ----------
        prob_fake, label, explanation = predict_deepfake(frames)
        prob_percent = round(prob_fake * 100.0, 2)

        # ---------- Chat summary ----------
        if label == "Fake":
            chat_summary = (
                f"I analyzed your {analysis_type} and estimate a "
                f"{prob_percent}% probability that it is a deepfake. "
                f"{explanation}"
            )
        else:
            chat_summary = (
                f"I analyzed your {analysis_type} and estimate only a "
                f"{prob_percent}% probability of deepfake. "
                f"{explanation}"
            )

        return jsonify(
            {
                "status": "ok",
                "label": label,
                "probability": prob_percent,
                "explanation": explanation,
                "chat_summary": chat_summary,
                "media_url": media_url,
                "analysis_type": analysis_type,
            }
        )

    except Exception as e:
        # print full stack trace in the server for debugging
        import traceback

        traceback.print_exc()
        # always return JSON so the frontend can show the message
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# -------------------------------------------------------------------
# 5. Simple chatbot endpoint
# -------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat

    Frontend sends JSON: {"message": "some text"}
    We respond with: {"reply": "assistant answer"}

    This is a very lightweight, rule-based bot to explain the model
    and give some guidance. You can replace this with a real NLP model.
    """
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify(
            {"reply": "Please type a question or upload a media file first."}
        )

    lower_msg = message.lower()

    # a few simple intent checks
    if "hello" in lower_msg or "hi" in lower_msg:
        reply = (
            "Hello! 👋 I'm the DeepGuard assistant. "
            "You can upload an image or video on the left and I’ll analyze it "
            "for deepfake traces. You can also ask me about the model or datasets."
        )
    elif "dataset" in lower_msg or "data set" in lower_msg:
        reply = (
            "For training deepfake detectors, common academic datasets include "
            "FaceForensics++, DFDC, and Celeb-DF. The model in this demo expects "
            "labeled real vs fake videos so it can learn temporal artifacts."
        )
    elif "how" in lower_msg and ("work" in lower_msg or "detect" in lower_msg):
        reply = (
            "I use an Xception CNN to extract features from each frame, then an "
            "LSTM to look at how those features change over time. Deepfakes often "
            "show subtle inconsistencies in textures, lighting, and motion that "
            "this combination can pick up."
        )
    elif "real" in lower_msg and "fake" in lower_msg:
        reply = (
            "Remember that no automated detector is perfect. A high fake score "
            "means the content looks suspicious and should be reviewed carefully, "
            "but it is not absolute proof. Always combine model output with "
            "human judgement and additional context."
        )
    else:
        reply = (
            "I specialize in deepfake detection. Try asking questions like "
            "'how does the model detect deepfakes?', 'which dataset can I use?', "
            "or just upload another media file for analysis."
        )

    return jsonify({"reply": reply})


# -------------------------------------------------------------------
# 6. Application entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    """
    When you run `python app.py`, this block executes.

    We set host to 127.0.0.1 so the app is only reachable from
    your own machine (not visible to other devices on the network).
    """
    app.run(
        host="127.0.0.1",  # local machine only (no 0.0.0.0 binding)
        port=5000,         # you can change this to 8000, 8080, etc.
        debug=True,        # shows nice error pages and auto-reloads
    )

    model = tf.keras.models.load_model(
    "models/xception_lstm.h5",
    compile=False
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

