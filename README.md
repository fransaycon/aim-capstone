# Exam Score Prediction - Capstone Project

A machine learning application that predicts student exam scores and provides personalized recommendations based on academic, behavioral, and contextual factors.

## 📋 Project Overview

This project uses a **Ridge Regression model** trained on 20,000 student records to predict exam performance (R² = 0.73, RMSE = 9.72). The system provides actionable, evidence-based recommendations for teachers to support student success.

**Key Features:**
- Predicts exam scores (0-100) based on student inputs
- Generates personalized recommendations using OpenAI GPT
- Simple web interface for easy data entry
- Real-time predictions via Flask API

## 📁 Project Structure

Workspace
(rerun without)
Collecting workspace information

aim-capstone/
├── app/
│ ├── backend/
│ │ ├── models/ # Trained model artifacts
│ │ │ ├── ridge_tuned_model.pkl
│ │ │ ├── scaler.pkl
│ │ │ ├── feature_names.json
│ │ │ └── encoding_info.json
│ │ ├── app.py # Flask API server
│ │ └── .env # Environment variables (OpenAI API key)
│ └── frontend/
│ ├── index.html # Web interface
│ ├── index.css # Styles
│ └── index.js # Client-side logic
├── slides/ # Project documentation & presentation
├── Franrey-Saycon-AIM-Capstone.ipynb # Model training notebook
├── kaggle-exam-score-prediction-dataset.csv
└── requirements.txt # Python dependencies


## 🚀 Installation & Setup

### Prerequisites

- **Python 3.13+** (minimum required)
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### 1. Clone the Repository

```bash
git clone <repository-url>
cd aim-capstone
```

### 2. Installation

```bash
# Create virtual env
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all requirements
pip install -r requirements.txt
```

### 3. Get your OPENAI API KEY

Create a .env file in app/backend with the content:
```
OPENAI_SECRET_KEY=your_openai_api_key_here
```

## 🏃 Running the Application

### 1. Start the Backend API
```bash
cd app/backend
python app.py
```

### 2. Start the Frontend (Static Server)
```bash
cd app/frontend
python -m http.server 8000
```

Access the app at http://localhost:8000

## 📚 Documentation
Full project documentation, EDA, model selection, and evaluation are available in:
Franrey-Saycon-AIM-Capstone.ipynb - Complete analysis notebook
slides - Presentation materials

### 🛠️ Technology Stack
Backend: Flask, scikit-learn, pandas, numpy
Frontend: Vanilla JavaScript, HTML5, CSS3
ML Model: Ridge Regression with StandardScaler
AI Integration: OpenAI GPT-4
Data: Kaggle Education Dataset (20,000 records)
