from flask import Flask, request, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# Reuse same clean function used in training
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    msg = data["message"]
    cleaned_msg = clean_text(msg)
    vec = vectorizer.transform([cleaned_msg])
    pred = model.predict(vec)[0]
    return jsonify({"prediction": "spam" if pred == 1 else "ham"})

if __name__ == "__main__":
    app.run(debug=True)
