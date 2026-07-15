# 🏭 Industrial Vision AI Quality Inspection System using Deep Learning

An end-to-end AI-powered manufacturing quality inspection platform that automatically detects casting defects using Deep Learning. The project combines a **PyTorch ResNet18 model**, **FastAPI backend**, and **Streamlit dashboard** to provide real-time defect detection, batch image inspection, analytics, and visualization.

---

## 🚀 Overview

Quality inspection is a critical stage in modern manufacturing. Manual inspection is often time-consuming, expensive, and susceptible to human error.

This project automates the inspection of casting components by leveraging transfer learning and computer vision. Users can upload one or multiple images, receive predictions in real time, analyze confidence scores, visualize results, and export reports.

---

## ✨ Features

* 🤖 Deep Learning based defect detection using ResNet18 Transfer Learning
* ⚡ FastAPI REST API for real-time inference
* 📊 Interactive Streamlit dashboard
* 📂 Batch image prediction
* 📈 Analytics dashboard with interactive Plotly charts
* 📜 Prediction history tracking
* 📥 CSV export of prediction results
* 📖 Auto-generated Swagger API documentation
* 🧩 Modular and production-style project architecture

---

## 🛠 Tech Stack

| Category                   | Technology                   |
| -------------------------- | ---------------------------- |
| Language                   | Python                       |
| Deep Learning              | PyTorch                      |
| Model                      | ResNet18 (Transfer Learning) |
| API                        | FastAPI                      |
| Dashboard                  | Streamlit                    |
| Data Processing            | Pandas                       |
| Visualization              | Plotly                       |
| Image Processing           | Pillow                       |
| Machine Learning Utilities | scikit-learn                 |

---

## 🏗 System Architecture

![Architecture](assets/architecture.png)

---

## 📸 Project Screenshots

### Dashboard

![Dashboard](assets/dashboard-home.png)

---

### Batch Prediction

![Batch Prediction](assets/batch-prediction.png)

---

### Analytics Dashboard

![Analytics](assets/analytics.png)

---

### Prediction History

![Prediction History](assets/prediction-history.png)

---

### FastAPI Swagger Documentation

![Swagger API](assets/swagger-api.png)

---

## 📂 Project Structure

```text
ai-manufacturing-defect-detection
│
├── app/
│   ├── api/
│   └── ml/
│
├── dashboard/
├── data/
├── docs/
├── models/
├── notebooks/
├── scripts/
├── tests/
├── assets/
├── sample_images/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## ⚙ Installation

### Clone the repository

```bash
git clone https://github.com/manojkumarv2508-cmyk/ai-manufacturing-defect-detection.git

cd ai-manufacturing-defect-detection
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Backend

```bash
uvicorn app.api.main:app --reload
```

Swagger documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

## 💻 Running the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📷 Sample Images

Use the images inside the **sample_images/** folder to quickly test the application.

The dashboard also supports uploading multiple images simultaneously for batch inspection.

---

## 📊 Model Performance

| Metric              |             Value |
| ------------------- | ----------------: |
| Model               |          ResNet18 |
| Training Strategy   | Transfer Learning |
| Classes             |    Defective / OK |
| Validation Accuracy |        **99.30%** |
| Inference           |         Real-time |
| Deployment          |     CPU Optimized |

---

## 🌐 REST API

### POST `/predict`

Uploads an image and returns:

```json
{
  "predicted_class": "def_front",
  "confidence_score": 1.0,
  "inference_time_seconds": 0.288,
  "message": "Success"
}
```

---

## 🔮 Future Improvements

* Multi-class defect detection
* YOLO-based object detection
* Live camera inspection
* Docker containerization
* Cloud deployment
* CI/CD pipeline
* Database integration
* Edge AI deployment

---

## 👨‍💻 Author

**Manoj Kumar V**

Artificial Intelligence & Machine Learning Engineer

* GitHub: https://github.com/manojkumarv2508-cmyk


---

## 📄 License

This project is licensed under the MIT License.
