import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def train_model():
    print("🔄 Training model...")
    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep="\t", names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Vectorize text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    # Train model
    model = MultinomialNB()
    model.fit(X, y)

    # Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("✅ Model trained and saved!")

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Loaded existing model.")
        return model, vectorizer
    else:
        print("⚠️ No model found. Training new one...")
        train_model()
        return load_model()

def predict_message(model, vectorizer):
    message = input("📩 Enter your message: ")
    X = vectorizer.transform([message])
    prediction = model.predict(X)[0]
    print("🚨 This is SPAM!" if prediction else "✅ This is NOT spam.")

def menu():
    model, vectorizer = load_model()
    while True:
        print("\n--- Spam Detector ---")
        print("1. Check Message")
        print("2. Retrain Model")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            predict_message(model, vectorizer)
        elif choice == "2":
            train_model()
            model, vectorizer = load_model()
        elif choice == "3":
            print("👋 Exiting...")
            break
        else:
            print("❌ Invalid choice!")

menu()
