import pickle
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from config import MODEL_FILE


def train_classifier(feedback_data):
    """Train classifier on manual labels (feedback dataset)"""
    if len(feedback_data) < 10:
        st.warning("Need at least 10 samples to train a meaningful classifier.")
        return None, None

    feedback_data.dropna(subset=['Question', 'Status'], inplace=True)
    feedback_data = feedback_data[feedback_data['Question'].str.len() > 0]

    valid_statuses = ["Accepted", "Rejected"]
    invalid_status_rows = ~feedback_data['Status'].isin(valid_statuses)
    if invalid_status_rows.any():
        st.error(f"Invalid status found in feedback data. Please correct: {feedback_data[invalid_status_rows]}")
        return None, None

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(feedback_data['Question'])
    y = feedback_data['Status'].apply(lambda x: 1 if x == "Accepted" else 0)

    if X.shape[0] != len(y):
        st.error(f"Mismatch in dataset size. X shape: {X.shape}, y length: {len(y)}. Please check for inconsistencies.")
        return None, None

    accepted_count = (y == 1).sum()
    rejected_count = (y == 0).sum()
    if accepted_count == 0 or rejected_count == 0:
        st.error("Need at least one 'Accepted' and one 'Rejected' question to train the classifier.")
        return None, None

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Save the model and vectorizer
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((vectorizer, model), f)

    return vectorizer, model


def load_classifier():
    """Load existing classifier from file"""
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                vectorizer, model = pickle.load(f)
            return vectorizer, model
        except:
            st.error("Failed to load the classifier model.")
            return None, None
    return None, None


def classify_questions(questions, vectorizer=None, model=None):
    """Classify questions using trained model"""
    if vectorizer is None or model is None:
        vectorizer, model = load_classifier()
        if vectorizer is None or model is None:
            st.warning("No trained classifier found. Please train the model first.")
            return []

    try:
        X_new = vectorizer.transform(questions)
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)[:, 1]  # Probability of being accepted

        results = []
        for q, pred, prob in zip(questions, predictions, probabilities):
            results.append({
                "Question": q,
                "Prediction": "Accepted" if pred == 1 else "Rejected",
                "Confidence": prob
            })

        return results
    except:
        st.error("Error classifying questions. The model might not be compatible with these questions.")
        return []