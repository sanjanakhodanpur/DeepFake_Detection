# 🤖 Deepfake Detection Web Application (Flask + XceptionNet)

This is a **Deepfake Detection Web Application** that allows users to upload **images or videos** and checks whether they are **real or fake (deepfake)** using an **AI model based on XceptionNet**.

The app uses **Flask** for the web interface and **TensorFlow** for deep learning inference.  
It provides a simple, elegant user interface built with HTML templates, shows a “Processing…” stage, and gives a detailed **result, confidence score, and explanation** for each uploaded file.

---

## 🧩 ABSTRACT

The increasing use of AI-generated fake videos and images (deepfakes) has made authenticity verification a major challenge.  
This project aims to identify such fake content by using a **deep learning model (XceptionNet)** trained to distinguish between real and AI-manipulated faces.

When a user uploads an image or video, the model analyzes it and predicts whether it is **real or fake**, displaying the confidence and a human-friendly explanation.

---

## 🎯 OBJECTIVE

- Detect and classify uploaded media (image or video) as **real** or **deepfake**.  
- Provide a simple and interactive **web-based interface** using Flask.  
- Generate confidence scores and clear explanations for model predictions.  
- Process both **images** and **videos** using the same system.

---

## 🧠 HOW THE SYSTEM WORKS

1. **User uploads a file** (image or video).  
2. The system verifies file type and saves it to `static/uploads/`.  
3. For **images**:
   - The image is preprocessed and analyzed using the XceptionNet-based deepfake detector.  
   - Output: “Real” or “Fake” + Confidence % + Explanation.  
4. For **videos**:
   - Frames are extracted using OpenCV.  
   - Each frame is analyzed individually, and an overall score is computed.  
   - Output: Overall video result + frame analysis summary.  
5. Flask renders the **result.html** page showing the result, confidence, timestamp, and explanation.

---

## 🧰 TECHNOLOGIES USED

| Category | Tool / Library |
|-----------|----------------|
| Backend Framework | Flask |
| Deep Learning | TensorFlow (XceptionNet model) |
| Image & Video Processing | OpenCV, Pillow, NumPy |
| Web Design | HTML, CSS, Bootstrap |
| Visualization | Matplotlib |
| Utility | Werkzeug (file uploads), datetime |
| Development | Visual Studio Code |

---

## 📁 FOLDER STRUCTURE


deepfake_detection/
│
├─ app.py → Main Flask application file
│
├─ models/
│ └─ deepfake_detector.py → Contains DeepfakeDetector class using XceptionNet
    __init_.py
│
├─ utils/
│ ├─ image_utils.py → Handles image loading and preprocessing
│ ├─ video_utils.py → Extracts frames from videos
__init_.py
│
├─ static/
│ ├─ uploads/ → Stores uploaded images and videos
│ ├─ css/, js/, logo.jpg → Optional frontend assets
│
├─ templates/
│ ├─ index.html → Home page (upload form)
│ ├─ result.html → Displays detection result
│ ├─ about.html, contact.html → Info pages
│
├─ requirements.txt → Python dependencies
└─ README.md → Project documentation

 INSTALLATION AND SETUP

### Step 1️⃣ — Create a Virtual Environment
```bash
python -m venv venv
Activate it:

Windows → venv\Scripts\activate

Mac/Linux → source venv/bin/activate

Step 2️ — Install Required Packages
bash
Copy code
pip install -r requirements.txt
If needed, you can manually install the main packages:

bash
Copy code
pip install flask tensorflow opencv-python pillow numpy werkzeug matplotlib
Step 3️ — Run the Flask Application
bash
Copy code
python app.py
If successful, you’ll see:

Loading AI models...
System ready!
 * Running on http://127.0.0.1:5000/
Open this address in your web browser.

 HOW TO USE
Go to http://127.0.0.1:5000/

Choose either:

Image Upload

 Video Upload


Then, the final result on the next page (result.html):

Real (Authentic)

 Fake (Deepfake Detected)

Confidence (0–100%)

Detailed explanation

SAMPLE OUTPUT
Input Type	Result	Confidence	Explanation
Image (face1.jpg)	Fake	89.2%	Detected texture and lighting irregularities
Video (sample.mp4)	Real	94.5%	Frames show consistent facial patterns
🧩 EXPLANATION GENERATION
Based on the model prediction:

Fake:

“Detected irregular facial features and digital artifacts.”

“Texture inconsistencies typical of AI-generated content.”

Real:

“Lighting, texture, and geometry appear natural.”

“No signs of manipulation detected.”

NOTES
Maximum upload size: 100 MB

Allowed formats:

Images → .jpg, .jpeg, .png

Videos → .mp4, .avi, .mov

The model loads once at startup for faster predictions.

If OpenCV cannot extract frames, the system flashes an error message.
