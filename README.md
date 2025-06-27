# End-to-End-Data-Science-Project
COMPANY: CODTECH IT SOLUTIONS

NAME: ANIKET BHOGE

INTERN ID:CT06DF1221

DOMAIN: DATA SCIENCE

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH
# 📧 Email Spam Detection using Machine Learning and Flask API

This project is an end-to-end solution that detects whether an email/SMS message is **spam** or **ham** (not spam) using Natural Language Processing and a Multinomial Naive Bayes classifier. The model is trained on the public **SMS Spam Collection Dataset** from Kaggle and deployed using a lightweight Flask API.

---

## 📌 Dataset Source
- UCI SMS Spam Collection Dataset  
- Kaggle Link: [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🚀 Project Pipeline

### 1. **Data Collection**
- Dataset loaded from `spam.csv`
- Columns used: `v1` (label), `v2` (message)

### 2. **Data Preprocessing**
- Lowercasing text
- Removing punctuation
- Removing stopwords using NLTK
- Mapping `ham` to 0 and `spam` to 1

### 3. **Feature Extraction**
- TF-IDF Vectorization of cleaned messages

### 4. **Model Building**
- Model: `Multinomial Naive Bayes`
- Training and evaluation using scikit-learn

### 5. **Model Evaluation**
- Classification Report
- Accuracy score
- Confusion Matrix

### 6. **Model Saving**
- `model.pkl` for classifier
- `vectorizer.pkl` for TF-IDF transformer

### 7. **API Deployment using Flask**
- `app.py` provides a `/predict` endpoint
- Accepts JSON: `{"message": "text"}`
- Returns: `{"prediction": "spam"}` or `"ham"`

---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/End-to-End-Data-Science-Project.git
cd End-to-End-Data-Science-Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
Run the notebook to retrain the model:
```bash
jupyter notebook task_3.ipynb
```

### 4. Run Flask API
```bash
python app.py
```

### 5. Test the API
```python
import requests
res = requests.post("http://127.0.0.1:5000/predict", json={"message": "Win a free trip now!"})
print(res.json())  # {'prediction': 'spam'}
```

---

## 📁 File Structure

```
End-to-End-Data-Science-Project/
├── app.py
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── task_3.ipynb
└── README.md
```

---

## 📚 Technologies Used
- Python
- Scikit-learn
- NLTK
- Flask
- Pandas, NumPy, Matplotlib
- Jupyter Notebook

---

