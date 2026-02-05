import pickle

# Load files
model = pickle.load(open("app/ml/model.pkl", "rb"))
vectorizer = pickle.load(open("app/ml/vectorizer.pkl", "rb"))

# Dummy test input (already preprocessed-style)
test_text = ["free cash win offer"]

# Vectorize
X = vectorizer.transform(test_text)

# Predict
print("Prediction:", model.predict(X))
