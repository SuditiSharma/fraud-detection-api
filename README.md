# 🔍 Fraud Detection API

A machine learning-powered REST API that detects fraudulent credit card transactions in real time.

Built with **FastAPI**, **scikit-learn**, and **Random Forest** — trained on 284,807 real-world transactions.

---

## 🎯 Background

This project was inspired by my experience working at **Aviva**, a FTSE 100 insurer, where I analysed claims data to identify patterns and anomalies. I wanted to combine that domain knowledge with my machine learning skills to build a production-ready fraud detection system.

---

## 🚀 Live Demo

> API deployed at: https://fraud-detection-api-mji8.onrender.com/docs

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Overall Accuracy | 100% |
| Fraud Precision | 84% |
| Fraud Recall | 83% |
| F1 Score (Fraud) | 83% |

Trained on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud cases.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **FastAPI** — REST API framework
- **scikit-learn** — Random Forest model
- **imbalanced-learn** — SMOTE for class imbalance
- **joblib** — model serialisation
- **uvicorn** — ASGI server
- **pandas / numpy** — data processing

---

## 📁 Project Structure
fraud-detection-api/
├── app/
│   ├── init.py
│   ├── main.py        # API endpoints
│   ├── model.py       # model loading and prediction
│   └── schemas.py     # request/response schemas
├── data/              # dataset (not tracked in git)
├── model/             # saved trained model
├── notebooks/         # exploratory analysis
├── train.py           # model training script
├── requirements.txt
└── README.md
---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/SuditiSharma/fraud-detection-api.git
cd fraud-detection-api
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the model**
```bash
python train.py
```

**5. Run the API**
```bash
uvicorn app.main:app --reload
```

**6. Open docs**
http://127.0.0.1:8000/docs
---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/predict` | POST | Predict fraud on a transaction |

---

## 📬 Example Request

```json
POST /predict
{
  "Time": 406.0,
  "V1": -2.31,
  "V2": 1.95,
  ...
  "Amount": 0.0
}
```

## ✅ Example Response

```json
{
  "prediction": 1,
  "probability_fraud": 0.92,
  "message": "⚠️ FRAUD DETECTED — 92.0% confidence"
}
```


---

## 👩‍💻 Author

**Suditi Sharma**
MSc Computer Science (Data Science) — University of Strathclyde
[LinkedIn](https://www.linkedin.com/in/suditi-sharma) | [GitHub](https://github.com/SuditiSharma)