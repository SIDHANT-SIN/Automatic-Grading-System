# Automatic-Grading-System
# Automated Grading System with OCR and Streamlit

An intelligent grading assistant that streamlines the evaluation process by using computer vision and OCR to extract roll numbers and total marks from answer sheet images. Built with Streamlit, OpenCV, and HuggingFace Transformers, this tool simplifies grading while ensuring accuracy and traceability.

---

## ✨ Features

- 📸 **Live Camera Capture**: Use your Android phone as an IP webcam to take answer sheet snapshots.
- 📁 **Image Upload Option**: Supports uploading `.jpg` images manually.
- 🧠 **Region Detection**: Automatically crops:
  - Roll number (digits)
  - Roll number (in words)
  - Total marks
- 🔍 **Handwriting OCR**: Uses `microsoft/trocr-large-handwritten` from HuggingFace for handwritten text recognition.
- 🎨 **Preprocessing Magic**: Uses CLAHE, Gaussian blur, and adaptive thresholding to clean up image noise.
- 🖥️ **Streamlit UI**: Clean, interactive interface for live grading.
- 🧾 **Cropped Output**: Saves cropped segments for manual verification if needed.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **OCR Engine**: HuggingFace Transformers (TrOCR)
- **Image Processing**: OpenCV, PIL
- **Others**: NumPy, Requests

---

## 📦 Installation

1. **Clone the Repository**:

```bash
git clone https://github.com/your-username/automated-grading-system.git
cd automated-grading-system
