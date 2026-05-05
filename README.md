# DeepGuard – Face Deepfake Detection Dashboard

DeepGuard is a web-based deepfake detection dashboard that analyzes **face images and videos** and estimates whether they are **real or fake**.  

The interface looks and feels like a modern SaaS application, with:

- Animated **“Analyzing media…”** overlay and progress steps  
- Side panel showing **prediction, confidence and explanation**  
- Built-in **chat assistant** that explains how the model works  
- Support for **image and video uploads** in the same UI

The backend is implemented with **Flask** and **TensorFlow**, using an **Xception + LSTM** model for temporal deepfake detection.

---

## 1. Tech Stack

- **Backend**
  - Python 3.10+  
  - Flask (web framework)  
  - TensorFlow 2.x (deep learning)  
  - OpenCV (video frame extraction)  
  - NumPy (numerical operations)

- **Frontend**
  - HTML5 templates (Jinja2 via Flask)  
  - CSS (custom design, modern dashboard theme)  
  - JavaScript (vanilla, no framework)  

---

## 2. Features

1. **Image & video analysis**
   - Upload **JPG/PNG** images or **MP4/AVI/MOV** videos.
   - Media is validated, stored in `static/uploads/` and passed to the detector.

2. **Xception + LSTM deepfake model**
   - Xception CNN extracts per-frame facial features.
   - LSTM models temporal patterns across frames.
   - Outputs a probability score (0–1) mapped to **Real / Fake**.

3. **Explainable output**
   - Returns a detailed text explanation based on the confidence score.
   - Describes artifacts such as spatial inconsistencies and temporal jitter.

4. **Interactive dashboard UI**
   - Hero “screen” with animated overlay and progress bar.
   - Side panel with prediction, confidence, reasoning.
   - Chat assistant panel on the right side.

5. **Chat assistant**
   - Simple rule-based chatbot (`/chat` endpoint).
   - Answers basic questions about:
     - Model architecture
     - Datasets for training
     - How predictions should be interpreted

---

## 3. Project Structure

```text
DEEPFAKE_detection/
│ app.py                      # Flask application
│ deepfake_detector.py        # Xception + LSTM model wrapper
│ video_utils.py              # Frame extraction & file helpers
│ requirements.txt            # Python dependencies
│ README.md                   # (this file)
│
├── models/
│   └── xception_lstm.h5      # Trained model weights (provided separately)
│
├── static/
│   ├── css/
│   │   └── style.css         # UI styling
│   ├── js/
│   │   └── main.js           # Frontend logic (upload + chat)
│   └── uploads/              # Uploaded images/videos (auto-created)
│
└── templates/
    ├── index.html            # Main dashboard + chatbot
    ├── about.html            # About page (architecture & motivation)
    ├── contact.html          # Contact / feedback page
    └── result.html           # Optional full result view
4. Requirements
4.1 System Requirements

OS: Windows 10/11, macOS, or Linux

Python: 3.10 or 3.11 (recommended)

RAM: 8 GB minimum (more recommended if running larger models)

Disk space: at least 3–4 GB free (TensorFlow + dependencies + model)

4.2 Python Packages

Defined in requirements.txt:
flask>=3.0.0,<4.0.0
tensorflow>=2.20.0,<2.21.0
opencv-python>=4.10.0
pillow>=10.0.0
numpy>=1.26.0

5. Installation Guide

Below is a step-by-step installation flow suitable for non-technical users.

Step 1 – Install Python

Install Python 3.10 or 3.11 from:
https://www.python.org/downloads/

During installation, check “Add Python to PATH”.

Step 2 – Download the project

Copy the entire project folder DEEPFAKE_detection to your machine
(for example: C:\Users\USERNAME\Desktop\DEEPFAKE_detection).

Make sure the structure matches the “Project Structure” section above.

Step 3 – Place the model weights

Ask the developer for the file: xception_lstm.h5.

Put it into:
DEEPFAKE_detection/models/xception_lstm.h5
If this file is missing, the app will still run, but predictions will be random (untrained model).

and also run 
-> python train_xception_lstm.py
the above will train the model in your laptop and then run the application that is python app.py but first go acc to the steps given and then run the application.

Step 5– Create a virtual environment

Open a terminal in the project folder:

On Windows (PowerShell)/ :
cd path\to\DEEPFAKE_detection

python -m venv venv
\venv\Scripts\Activate

Step 4 – Install Python dependencies

With the virtual environment activated:

pip install Flask, TensorFlow, OpenCV, Pillow and NumPy.

pip install --upgrade pip
pip install -r requirements.txt


This will install Flask, TensorFlow, OpenCV, Pillow and NumPy.

6. Running the Application

With the virtual environment activated and dependencies installed:

python app.py


If everything is correct, you should see something like:

 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)


Now open your browser and go to:

http://127.0.0.1:5000


You will see the DeepGuard dashboard.
7. Using the Application
7.1 Main Dashboard (Home)

Upload media

Click “Upload media” on the left hero section.

Choose an image (.jpg, .jpeg, .png) or a video (.mp4, .avi, .mov).

Preview

The selected media will appear in the mock laptop screen:

Images display as a still picture.

Videos display with playback controls.

Analysis

As soon as a file is selected, the app will:

Show an “Analyzing media…” overlay with:

Animated progress bar

“Face alignment → Feature extraction → Temporal analysis → Final verdict” steps

Send the file to the backend (/analyze API).

Results

When analysis completes:

The overlay disappears.

The side panel on the right of the mock screen updates with:

Prediction: Real or Fake

Confidence (%): probability of deepfake

Media type: Image or Video

Reasoning: textual explanation of the decision.

A summary of the result is also posted automatically in the chat panel.

7.2 Chat Assistant

Located on the right side of the screen.

You can:

Ask: “How do you detect deepfakes?”

Ask: “Which dataset did you train on?”

Ask: “What does a fake score mean?”

The assistant is currently rule-based (no external API calls), so it is safe to run offline.