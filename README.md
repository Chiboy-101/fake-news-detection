# 📰 Fake News Detection using Bi-directional LSTM

A deep learning project that classifies news articles as **Fake** or **Real** using a Bi-directional LSTM model.  
The model achieves a final accuracy of **~97%** ✅ and includes a **FastAPI deployment** for real-time predictions.

---

## 📈 Model Results
| Metric                     | Accuracy |
| -------------------------- | -------- |
| **Training Accuracy**     | ~97%     |
| **Validation Accuracy**  | ~96%     |
| **Test Accuracy**        | ~97%     |

The model demonstrates strong generalization on unseen data, making it reliable for real-world news classification.

---

## 🔍 Project Overview

Fake news is a growing problem in the digital age. This project implements a **deep learning-based solution** using a **Bi-directional LSTM** network to detect whether a news article is Fake or Real.  

The model learns patterns from news content and can be accessed via a **FastAPI** endpoint for real-time predictions.

---

## ✨ Features

- Text preprocessing including **tokenization** and **padding** 
- Combines **title** and **text** for better feature representation 
- **Bi-directional LSTM** for sequence learning 
- Dropout layers to reduce overfitting 
- Real-time **FastAPI deployment** for predictions  
- Supports testing using CSV datasets 

---

## 📊 Dataset

The dataset is **not included** due to size constraints.  

- **Main dataset:** `WELFake_Dataset.csv`  
  - Columns used: `title`, `text`, `label`  
  - Labels: `0` = Real, `1` = Fake  
- **Test datasets:** `True.csv`, `Fake.csv` (optional for evaluation)

### How to get the dataset:

1. Download the WELFake dataset from [Kaggle](https://www.kaggle.com/datasets)  
2. Place the CSV files inside the `data/` folder 📁  

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

#Train the model
python src/train_model.py
# Save both trained model and tokenizer in /models

# Run the FastAPI Server
uvicorn src/main:app --reload
# Open http://127.0.0.1:8000/docs for interactive API documentation.