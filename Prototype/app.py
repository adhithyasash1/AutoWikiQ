"""
At its core, this project is all about closing the loop between human judgment and automated question generation so that, over time, you can reliably produce high‑quality, research‑style questions from any Wikipedia topic with minimal manual effort. Here’s what each piece “earns” you and what you end up with:

Human‑in‑the‑Loop Data Collection (Modes 1 & 2)
Purpose: Gather a labeled dataset of “Accepted” vs. “Rejected” questions so the system knows what a good question looks like.
Outcome: You accumulate a CSV of real human feedback, giving you a foundation to teach a machine what qualities make questions acceptable.
Training the Classifier (Mode 2)
Purpose: Turn those human labels into a predictive model (TF‑IDF + Random Forest) that can automatically judge future questions.
Outcome: You end up with a saved vectorizer + model combo that can, given any candidate question, output “Accepted” or “Rejected” (with a confidence score) in milliseconds.
Automated Question Generation & Filtering (Mode 3)
Purpose: Leverage the LLM to propose batches of questions, then use your classifier to filter them until you’ve got N that pass your confidence threshold.
Outcome: A repeatable “6‑step pipeline” that, for any new topic, spins up, generates, filters, and logs exactly N high‑confidence questions without you having to eyeball each one.
Efficiency Tracking & Analytics (Mode 4)
Purpose: Quantify how many LLM calls you need per accepted question and compare it against pure manual review.
Outcome: A plot of manual vs. automated acceptance rates and a running log of your system’s improving efficiency—so you can demonstrate, “Hey, by version 3.2 of my model, I only needed 1.8 calls per accepted question.”
"""
import streamlit as st
import pandas as pd
import wikipediaapi
# import ollama
import re
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime
from google import genai
import random

# A small collection of slightly varied prompts for retrying when duplication occurs
PROMPT_VARIANTS = [
    "Based on the content below, generate exactly {n} clear and concise questions. Focus on academic & research angles:\n\n{text}",
    "Please produce {n} precise and scholarly questions from the following text:\n\n{text}",
    "From the excerpt below, craft exactly {n} rigorous academic questions:\n\n{text}"
]

# ←←← CONFIGURE YOUR GEMINI KEY & MODEL →→→
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
if not GENAI_API_KEY:
    st.error("🚨 GEMINI_API_KEY not set! Run `export GEMINI_API_KEY=...` and restart.")
    st.stop()
genai_client = genai.Client(api_key=GENAI_API_KEY)  # Initialize once

# ─── Streamlit page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Wiki Question Generator", layout="centered")
st.title("Wikipedia Question Generator with Feedback")


# ─── Disk I/O helpers ───────────────────────────────────────────────────────────────────────

# Load existing feedback data if available
def load_feedback_data():
    filename = "questions_feedback.csv"
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return pd.read_csv(filename)
    return pd.DataFrame(columns=['Question', 'Status', 'Timestamp'])


# Load acceptance rate history
def load_acceptance_history():
    filename = "acceptance_history.csv"
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        df = pd.read_csv(filename)
        # Ensure "N" column exists
        if "N" not in df.columns:
            df["N"] = np.nan
        # Cast dtypes for consistency
        df = df.astype({"Samples": float, "Manual": float, "Automated": float, "N": float})
        return df
    return pd.DataFrame(columns=["Samples", "Manual", "Automated", "N"])


# Load classified questions
def load_classified_questions():
    filename = "classified_questions.csv"
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return pd.read_csv(filename)
    return pd.DataFrame(columns=['id', 'question', 'predicted_status', 'confidence', 'timestamp', 'topic'])


# Helper to Compute the accuracy of the classifier
def get_true_labels(questions):
    """
    Given a list of question strings, look up their human 'Accepted'/'Rejected'
    status in st.session_state.feedback_data and return [0|1] for each.
    If a question is not found, we skip it.
    """
    fb = st.session_state.feedback_data.set_index('Question')['Status'].to_dict()
    true = []
    for q in questions:
        if q in fb:
            true.append(1 if fb[q] == 'Accepted' else 0)
    return true


# ─── Session-state init ────────────────────────────────────────────────────────────────────

# Initialize session state variables
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = load_feedback_data()
if 'acceptance_history' not in st.session_state:
    st.session_state.acceptance_history = load_acceptance_history()
if 'classified_questions' not in st.session_state:
    st.session_state.classified_questions = load_classified_questions()
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'checkbox_states' not in st.session_state:
    st.session_state.checkbox_states = {}
if 'classified_results' not in st.session_state:
    st.session_state.classified_results = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'classifier_trained' not in st.session_state:
    st.session_state.classifier_trained = False
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = 5
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Generate"  # "Generate" or "Classify"
if 'run_offset' not in st.session_state:
    st.session_state.run_offset = 0


# ─── Fetch + generate via Gemini & Wikipedia ───────────────────────────────────────────────

# Wikipedia Topic Input
def fetch_wikipedia_content(topic):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='YourAppName/1.0 (your_email@example.com)', language="en")
    try:
        page = wiki_wiki.page(topic)
        if not page.exists(): return None
        return page.text[:5000]  # Limit text size to avoid overloading
    except Exception as e:
        st.error(f"Wikipedia fetch failed: {e}")
        return None

    # Generate questions using google's gemini api


def generate_questions_from_text(text, num_questions, variant=False):
    if variant:
        template = random.choice(PROMPT_VARIANTS)
        prompt = template.format(n=num_questions, text=text[:1500])
    else:
        prompt = f"Based on the following content, generate exactly {num_questions} clear and concise questions. Only academic & research based questions.\n\n{text[:1500]}"
    try:
        resp = genai_client.models.generate_content(
            model=GENAI_MODEL,
            contents=prompt
        )
        raw_output = resp.text
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return []

    # parse numbered or bullet lines into question strings
    possible = re.findall(
        r'(?:\d+\.\s*|[-*]\s*|^)(.*?)(?=\n|$)',
        raw_output.strip(),
        re.MULTILINE
    )
    questions = [
        q.strip() for q in possible
        if q.strip() and (
                '?' in q or
                q.lower().startswith(('what', 'how', 'why'))
        )
    ]
    return questions[:num_questions]


# ─── Train classifier on manual labels (feedback dataset) ──────────────────────────────────────────────────────

def train_classifier():
    if len(st.session_state.feedback_data) < 10:
        st.warning("Need at least 10 samples to train a meaningful classifier.")
        return None, None

    st.session_state.feedback_data.dropna(subset=['Question', 'Status'], inplace=True)
    # remove the empty strings
    st.session_state.feedback_data = st.session_state.feedback_data[
        st.session_state.feedback_data['Question'].str.len() > 0]
    # Validate Status values
    valid_statuses = ["Accepted", "Rejected"]
    invalid_status_rows = ~st.session_state.feedback_data['Status'].isin(valid_statuses)
    if invalid_status_rows.any():
        st.error(
            f"Invalid status found in feedback data. Please correct: {st.session_state.feedback_data[invalid_status_rows]}")
        return None, None

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(st.session_state.feedback_data['Question'])
    y = st.session_state.feedback_data['Status'].apply(lambda x: 1 if x == "Accepted" else 0)

    if X.shape[0] != len(y):
        st.error(f"Mismatch in dataset size. X shape: {X.shape}, y length: {len(y)}. Please check for inconsistencies.")
        return None, None

    # Check for minimum class representation
    accepted_count = (y == 1).sum()
    rejected_count = (y == 0).sum()
    if accepted_count == 0 or rejected_count == 0:
        st.error("Need at least one 'Accepted' and one 'Rejected' question to train the classifier.")
        return None, None

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Save the model and vectorizer
    with open("question_classifier.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)

    return vectorizer, model


# ─── Classify (new questions) via trained model ────────────────────────────────────────────────────────────────────────────

def classify_questions(questions):
    # Try to load existing model or use session state model
    vectorizer = None
    model = None

    if st.session_state.vectorizer is not None and st.session_state.model is not None:
        vectorizer = st.session_state.vectorizer
        model = st.session_state.model
    elif os.path.exists("question_classifier.pkl"):
        try:
            with open("question_classifier.pkl", "rb") as f:
                vectorizer, model = pickle.load(f)
                st.session_state.vectorizer = vectorizer
                st.session_state.model = model
        except:
            st.error("Failed to load the classifier model.")
            return []
    else:
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


# ─── Save classified questions to CSV ───────────────────────────────────────────────────────────────

def save_classified_questions(results, topic):
    """Save the classified questions to a CSV file"""
    filename = "classified_questions.csv"

    # Create a DataFrame with the classification results
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = []
    for result in results:
        data.append({
            'question': result['Question'],
            'predicted_status': result['Prediction'],
            'confidence': round(result['Confidence'], 2),
            'timestamp': now,
            'topic': topic
        })

    df = pd.DataFrame(data)

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0

    # Get the next ID value
    if file_exists:
        existing_df = pd.read_csv(filename)
        next_id = existing_df['id'].max() + 1 if not existing_df.empty else 1
    else:
        next_id = 1

    # Add ID column
    df.insert(0, 'id', range(next_id, next_id + len(df)))

    # Append to existing file or create new one
    df.to_csv(filename, mode='a', header=not file_exists, index=False)

    # Update session state
    st.session_state.classified_questions = load_classified_questions()

    return len(data)


# ─── Helper Functions for 6 Step Pipeline and Analytics Modes ────────────────────────────────

def run_pipeline_mode(topic: str, N: int, threshold: float):
    """
    Executes the 6-step pipeline for generating and classifying questions for a given topic.
    Handles prompt retry, classification, result persistence, acceptance rate computation,
    history updating, and UI feedback. Mirrors Step 3's prior logic.
    """
    # Retrieve previous automated acceptance rates
    prev_autos = st.session_state.acceptance_history["Automated"].dropna()
    last_auto = prev_autos.iloc[-1] if not prev_autos.empty else None
    with st.spinner("Running the 6-step pipeline…"):
        # Try up to two attempts: first default prompt, then a variant if needed
        for attempt in (False, True):
            # (re-)initialize run state
            all_res = []
            M = 0

            # Step 0–1: fetch & initial generate
            text = fetch_wikipedia_content(topic)
            if not text:
                st.error("Topic not found on Wikipedia.")
                st.stop()
            initial_qs = generate_questions_from_text(text, N, variant=attempt)
            initial_res = classify_questions(initial_qs)
            all_res += initial_res
            M += len(initial_res)

            # Step 2–4: top‑up until we have N accepted
            accepted = [r for r in all_res if r["Confidence"] >= threshold]
            while len(accepted) < N:
                to_gen = N - len(accepted)
                extra_qs = generate_questions_from_text(text, to_gen, variant=attempt)
                extra_res = classify_questions(extra_qs)
                all_res += extra_res
                M += len(extra_res)
                accepted = [r for r in all_res if r["Confidence"] >= threshold]

            # Step 5: select top N
            final_accepted = sorted(accepted, key=lambda r: r["Confidence"], reverse=True)[:N]

            # Persist results for later analysis
            save_classified_questions(all_res, topic)
            # compute auto efficiency
            auto_rate = N / M * 100

            # If this is the first attempt and auto_rate is unchanged, retry with variant
            if attempt is False and last_auto is not None and auto_rate == last_auto:
                st.warning("Automated acceptance rate unchanged; retrying with a slightly tweaked prompt…")
                continue

            # Otherwise, break out and proceed with this attempt's results
            break

        # show final bucket
        st.success(f"Final {len(final_accepted)} accepted questions out of {M} generated")
        for r in final_accepted:
            st.write(f"- {r['Question']}  (conf={r['Confidence']:.2f})")

        # —— compute manual acceptance on *this* run’s slice ——
        fb = st.session_state.feedback_data.sort_values("Timestamp").reset_index(drop=True)
        start = st.session_state.run_offset
        end = start + M  # cumulative total so far
        segment = fb.iloc[start:end]  # slice this run’s questions
        manual_seg = segment["Status"].eq("Accepted").mean() * 100 if len(segment) > 0 else 0
        # now bump run_offset for next time
        st.session_state.run_offset = end

        st.info(f"Automated acceptance rate: **{auto_rate:.1f}%**    "
                f"(Manual over this run: {manual_seg:.1f}%)")

        # append *only* this run
        new_run = pd.DataFrame({
            "Samples": [end],
            "Manual": [manual_seg],
            "Automated": [auto_rate],
            "N": [N]
        }).dropna(subset=["Automated"])
        hist = pd.concat([st.session_state.acceptance_history, new_run], ignore_index=True)
        # drop only fully duplicate rows (i.e. identical Samples, Manual, Automated, and N)
        hist = hist.drop_duplicates(keep="last")
        st.session_state.acceptance_history = hist
        hist.to_csv("acceptance_history.csv", index=False)

        # batch accuracy on final_accepted vs human labels
        true_labels = get_true_labels([r["Question"] for r in final_accepted])
        if true_labels:
            acc = accuracy_score(true_labels, [1] * len(true_labels))
            st.info(f"Classifier accuracy on overlap: **{acc * 100:.1f}%**")
        else:
            st.warning("No overlap with manual labels → cannot compute batch accuracy.")


def plot_analytics_mode():
    """
    Plots the manual vs automated acceptance rate analytics for pipeline runs.
    Loads run history, computes cumulative efficiency, and displays a matplotlib plot.
    Mirrors the prior analytics block exactly, using st.session_state.
    """
    hist = st.session_state.acceptance_history.copy()
    fb = st.session_state.feedback_data

    if hist.empty:
        st.warning("No pipeline runs yet—run Test Classification first.")
    else:
        # prepend (0,0) baseline if missing
        if hist["Samples"].min() > 0:
            zero = pd.DataFrame({
                "Samples": [0],
                "Manual": [0.0],
                "Automated": [0.0],
                "N": [0]
            })
            hist = pd.concat([zero, hist], ignore_index=True)

        hist = hist.sort_values("Samples").reset_index(drop=True)

        # per-run x and y values
        x = hist["Samples"]
        y_man = hist["Manual"]
        y_auto = hist["Automated"]

        # ─── Compute cumulative efficiency ────────────────────────────
        # cumulative accepted questions
        cum_accepted = hist["N"].cumsum()
        # avoid division by zero
        valid_samples = hist["Samples"].where(hist["Samples"] != 0, np.nan)
        cum_eff = cum_accepted.div(valid_samples) * 100
        cum_eff = cum_eff.fillna(0)

        fig, ax = plt.subplots(figsize=(30, 15))
        ax.plot(x, y_man, "-o", label="Manual acceptance (segment)")
        ax.plot(x, y_auto, "--s", label="Auto efficiency (per run)")
        ax.plot(x, cum_eff, ":d", label="Cumulative efficiency (i·N/ΣM)")
        ax.set_xlabel("Total LLM calls (cumulative M)")
        ax.set_ylabel("Acceptance Rate (%)")
        ax.set_title("Manual vs Automated Acceptance Rate per Run")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

        # annotate per-run and cumulative
        for i, (xi, mi, ai) in enumerate(zip(x, y_man, y_auto)):
            if i % 2 == 0:
                ax.annotate(f"{mi:.1f}%", (xi, mi),
                            xytext=(0, +8), textcoords="offset points",
                            ha="center", color="tab:blue")
                ax.annotate(f"{ai:.1f}%", (xi, ai),
                            xytext=(0, -12), textcoords="offset points",
                            ha="center", color="tab:orange")
                ax.annotate(f"{cum_eff.iloc[i]:.1f}%", (xi, cum_eff.iloc[i]),
                            xytext=(0, -28), textcoords="offset points",
                            ha="center", color="tab:green")

        st.pyplot(fig)


# ─── Sidebar + modes ──────────────────────────────────────────────────────────────────────────────────────

# UI Mode Selection
st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Select Mode",
                        ["Generate & Feedback", "Train Classifier", "Test Classification", "View Analytics"])
st.session_state.current_mode = mode

# users can tune how the automated loop decides “accepted”
threshold = st.sidebar.slider(
    "Auto-Accept Confidence Threshold",
    min_value=0.0, max_value=1.0, value=0.7,
    help="When auto-selecting questions, only those ≥ this confidence count."
)

# Mode 1: Generate Questions and Collect Feedback (Task 1)
if mode == "Generate & Feedback":
    st.header("Step 1: Generate Questions and Collect Feedback")
    st.info(
        "This step collects feedback data to train the classifier. Try to generate at least 100 questions and provide feedback.")

    topic = st.text_input("Enter a Wikipedia topic to generate questions:")
    st.session_state.num_questions = st.number_input("Number of questions to generate:", min_value=1, max_value=20,
                                                     value=5)

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


# Mode 2: Train Classifier (Task 2)
elif mode == "Train Classifier":
    st.header("Step 2: Train Question Classifier")
    st.info(
        "This step trains a machine learning model on your feedback data to predict which questions are likely to be accepted.")

    # Display current dataset size
    if not st.session_state.feedback_data.empty:
        total = len(st.session_state.feedback_data)
        accepted = st.session_state.feedback_data['Status'].value_counts().get("Accepted", 0)
        st.info(f"Current dataset: {total} questions ({accepted} accepted, {total - accepted} rejected)")

        if total < 10:
            st.warning("You need at least 10 labeled questions to train a classifier. Please go back to Step 1.")
        else:
            if st.button("Train Classifier Model"):
                with st.spinner("Training model..."):
                    vectorizer, model = train_classifier()
                    if vectorizer is not None and model is not None:
                        st.session_state.vectorizer = vectorizer
                        st.session_state.model = model
                        st.session_state.classifier_trained = True
                        st.success("Classifier trained and saved successfully!")
                        st.info("Now you can proceed to Step 3 to test the classifier.")
                    else:
                        st.error("Failed to train classifier.")

            # Display model statistics if available
            if os.path.exists("question_classifier.pkl"):
                st.success("A trained model is available.")

                # Show model's feature importance if we have a trained model
                if st.session_state.model is not None and st.session_state.vectorizer is not None:
                    with st.expander("Show Model Feature Importance"):
                        if hasattr(st.session_state.model, 'feature_importances_'):
                            importances = st.session_state.model.feature_importances_
                            feature_names = st.session_state.vectorizer.get_feature_names_out()

                            # Get top 15 features
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

# ─── Mode 3: Test Classification (6-Step Pipeline) ───────────────────────────────────
elif mode == "Test Classification":
    # Delegated to run_pipeline_mode
    st.header("Step 3: Test Question Classification")
    st.info(
        "This step uses the 6-step pipeline to generate N accepted questions and compute an automated acceptance rate.")
    if st.session_state.classifier_trained:
        topic = st.text_input("Enter a Wikipedia topic to generate and classify questions:")
        N = st.number_input("Target # of accepted questions (N):", 1, 20, value=5)
        if topic and st.button("Run Pipeline"):
            run_pipeline_mode(topic, N, threshold)
    else:
        st.warning("No trained classifier available. Please complete Step 2 first.")


# ─── Mode 4: View Analytics (per-run N/M efficiency) ────────────────────────────────
elif mode == "View Analytics":
    # Delegated to plot_analytics_mode
    st.header("Manual vs Automated Acceptance Rate per Pipeline Run")
    st.info("Each point shows one run’s efficiency: N/M and human acceptance in that segment.")
    plot_analytics_mode()
else:
    st.warning("No feedback data available. Please generate questions and provide feedback in Step 1 first.")