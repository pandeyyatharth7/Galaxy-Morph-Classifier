# 🔭 Galaxy Morph — A Machine Learning Approach to Galaxy Shape Classification

An AI-powered system that classifies galaxy morphologies from deep-space images using a hybrid **Neural Network + Symbolic Reasoning** architecture.

<!-- 
## 🌐 Live Demo
- **Frontend:** [galaxy-morph.vercel.app](https://galaxy-morph.vercel.app)  
- **Backend API:** [galaxy-morph-api.onrender.com](https://galaxy-morph-api.onrender.com)
-->

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Transfer Learning** | MobileNetV2 backbone pre-trained on 1.4M ImageNet images |
| 🔍 **Grad-CAM Explainability** | Visual heatmaps showing where the AI focuses |
| ⚡ **Symbolic Reasoning** | Rule-based forward chaining with confidence scoring |
| 🔄 **Human-in-the-Loop Learning** | Expert feedback adjusts AI's internal rule confidences in real-time |
| 📦 **Batch Classifier** | Autonomous bulk processing of entire image folders |
| 🌐 **Full-Stack Web App** | Next.js frontend + FastAPI backend |

---

## 🏗️ Architecture

```
📸 Galaxy Image
     │
     ▼
┌─────────────────────────────┐
│  Vision System (Vision.py)  │   ← MobileNetV2 Transfer Learning
│  Feature Extraction (CNN)   │
└────────────┬────────────────┘
             │ Percepts: P_EL, P_CW, P_ACW, EDGE_ON, MERGER
             ▼
┌─────────────────────────────┐
│  Reasoning Engine (Src.py)  │   ← Symbolic Forward Chaining
│  IF-THEN Rules + Beliefs    │
└────────────┬────────────────┘
             │ Belief State + Decision
             ▼
┌─────────────────────────────┐
│  Decision Module            │   ← Rational Agent (max belief)
│  Final Classification       │
└────────────┬────────────────┘
             │
             ▼
     🏷️ Elliptical / Spiral / Uncertain
             │
             ▼ (optional)
┌─────────────────────────────┐
│  Learning Module            │   ← Human-in-the-Loop Feedback
│  Confidence Adjustment      │
└─────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | TensorFlow / Keras + MobileNetV2 |
| **Backend API** | Python, FastAPI, Uvicorn |
| **Frontend** | Next.js 16, React 19, TypeScript |
| **Data** | Galaxy Zoo 1 (SDSS DR7) |
| **Explainability** | Grad-CAM |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start API server
python -m uvicorn api:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📁 Project Structure

```
├── api.py                  # FastAPI backend (predict + feedback endpoints)
├── Vision.py               # MobileNetV2 model architecture
├── Src.py                  # Symbolic reasoning engine + learning module
├── Train_Vision.py         # Model training script
├── Evaluate.py             # Model evaluation
├── BatchClassifier.py      # Autonomous batch classification tool
├── Get_Images.py           # Galaxy image downloader (SDSS)
├── app.py                  # Streamlit UI (alternative frontend)
├── galaxy_eye.weights.h5   # Trained model weights
├── requirements.txt        # Python dependencies
└── frontend/               # Next.js web application
    └── src/app/
        └── page.tsx        # Main UI with upload, Grad-CAM, feedback
```

## 📜 License

This project is open for learning and research purposes.

---

## 👤 Author

Built with 🔭 by Yatharth Pandey