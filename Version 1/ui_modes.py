import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from question_generator import fetch_wikipedia_content, generate_questions_from_text
from classifier import train_classifier, load_classifier
from pipeline import run_pipeline_mode
from analytics import plot_analytics_mode
from config import MODEL_FILE


def mode_generate_feedback():
    """Mode 1: Generate Questions and Collect Feedback"""
    st.header("Step 1: Generate Questions and Collect Feedback")
    st.info(
        "This step collects feedback data to train the classifier. Try to generate at least 100 questions and provide feedback.")

    topic = st.text_input("Enter a Wikipedia topic to generate questions:")
    st.session_state.num_questions = st.number_input("Number of questions to generate:",
                                                     min_value=1, max_value=20, value=5)

    if topic and st.button("Fetch and Generate Questions"):
        with st.spinner("Fetching Wikipedia content and generating questions..."):
            text = fetch_wikipedia_content(topic)
            if text:
                st.session_state.questions = generate_questions_from_text(text, st.session_state.num_questions)
                if len(st.session_state.questions) < st.session_state.num_questions:
                    st.warning(f"Only generated {len(st.session_state.questions)} valid questions.")
                st.session_state.checkbox_states = {f"q_{idx}": False for idx in range(len(st.session_state.questions))}
            else:
                st.error("Topic not found on Wikipedia.")

    if st.session_state.questions:
        st.subheader("Review Generated Questions:")
        for idx, question in enumerate(st.session_state.questions):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"**{idx + 1}. {question}**")
            with col2:
                st.session_state.checkbox_states[f"q_{idx}"] = st.checkbox(
                    "Accept", key=f"q_{idx}", value=st.session_state.checkbox_states[f"q_{idx}"]
                )

        if st.button("Submit Feedback"):
            feedback_list = [
                {
                    'Question': question,
                    'Status': "Accepted" if st.session_state.checkbox_states[f"q_{idx}"] else "Rejected",
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                for idx, question in enumerate(st.session_state.questions)
            ]
            st.session_state.feedback_data = pd.concat(
                [st.session_state.feedback_data, pd.DataFrame(feedback_list)], ignore_index=True
            )
            st.session_state.feedback_data.to_csv("questions_feedback.csv", index=False)
            st.success(f"Feedback submitted for {len(st.session_state.questions)} questions!")
            st.session_state.questions = []

            # Show current dataset stats
            total = len(st.session_state.feedback_data)
            accepted = st.session_state.feedback_data['Status'].value_counts().get("Accepted", 0)
            st.info(f"Current dataset: {total} questions ({accepted} accepted, {total - accepted} rejected)")

            if total >= 100:
                st.success(
                    "You've reached 100+ labeled questions! You can now proceed to Step 2 to train the classifier.")


def mode_train_classifier():
    """Mode 2: Train Classifier"""
    st.header("Step 2: Train Question Classifier")
    st.info(
        "This step trains a machine learning model on your feedback data to predict which questions are likely to be accepted.")

    if not st.session_state.feedback_data.empty:
        total = len(st.session_state.feedback_data)
        accepted = st.session_state.feedback_data['Status'].value_counts().get("Accepted", 0)
        st.info(f"Current dataset: {total} questions ({accepted} accepted, {total - accepted} rejected)")

        if total < 10:
            st.warning("You need at least 10 labeled questions to train a classifier. Please go back to Step 1.")
        else:
            if st.button("Train Classifier Model"):
                with st.spinner("Training model..."):
                    vectorizer, model = train_classifier(st.session_state.feedback_data)
                    if vectorizer is not None and model is not None:
                        st.session_state.vectorizer = vectorizer
                        st.session_state.model = model
                        st.session_state.classifier_trained = True
                        st.success("Classifier trained and saved successfully!")
                        st.info("Now you can proceed to Step 3 to test the classifier.")
                    else:
                        st.error("Failed to train classifier.")

            if os.path.exists(MODEL_FILE):
                st.success("A trained model is available.")

                if st.session_state.model is not None and st.session_state.vectorizer is not None:
                    with st.expander("Show Model Feature Importance"):
                        if hasattr(st.session_state.model, 'feature_importances_'):
                            importances = st.session_state.model.feature_importances_
                            feature_names = st.session_state.vectorizer.get_feature_names_out()

                            indices = np.argsort(importances)[::-1][:15]
                            top_features = [(feature_names[i], importances[i]) for i in indices]

                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh([f[0] for f in top_features], [f[1] for f in top_features])
                            ax.set_xlabel("Feature Importance")
                            ax.set_title("Top 15 Important Features for Classification")
                            st.pyplot(fig)
                        else:
                            st.info("Feature importance not available for this model type.")
    else:
        st.warning("No feedback data available. Please generate questions and provide feedback in Step 1 first.")


def mode_test_classification(threshold):
    """Mode 3: Test Classification"""
    st.header("Step 3: Test Question Classification")
    st.info(
        "This step uses the 6-step pipeline to generate N accepted questions and compute an automated acceptance rate.")

    if st.session_state.classifier_trained:
        topic = st.text_input("Enter a Wikipedia topic to generate and classify questions:")
        N = st.number_input("Target # of accepted questions (N):", 1, 20, value=5)
        if topic and st.button("Run Pipeline"):
            run_pipeline_mode(topic, N, threshold, st.session_state)
    else:
        st.warning("No trained classifier available. Please complete Step 2 first.")


def mode_view_analytics():
    """Mode 4: View Analytics"""
    st.header("Manual vs Automated Acceptance Rate per Pipeline Run")
    st.info("Each point shows one run's efficiency: N/M and human acceptance in that segment.")
    plot_analytics_mode(st.session_state)