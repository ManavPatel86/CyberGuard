import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------- Safe NLTK setup ----------
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

ps = PorterStemmer()

# ---------- Load model & vectorizer safely ----------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# ---------- SAME preprocessing logic as CampusX ----------
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = text.split()

    cleaned = []
    for i in tokens:
        if i.isalnum() and i not in STOPWORDS:
            cleaned.append(ps.stem(i))

    return " ".join(cleaned)

# ---------- Prediction ----------
def predict_spam(text: str) -> int:
    transformed = transform_text(text)
    vector = vectorizer.transform([transformed])
    prediction = model.predict(vector)[0]
    return int(prediction)
