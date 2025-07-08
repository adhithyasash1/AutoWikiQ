import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import FEEDBACK_FILE, HISTORY_FILE, CLASSIFIED_FILE


def load_feedback_data():
    """Load existing feedback data if available"""
    if os.path.exists(FEEDBACK_FILE) and os.path.getsize(FEEDBACK_FILE) > 0:
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=['Question', 'Status', 'Timestamp'])


def load_acceptance_history():
    """Load acceptance rate history"""
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        df = pd.read_csv(HISTORY_FILE)
        if "N" not in df.columns:
            df["N"] = np.nan
        df = df.astype({"Samples": float, "Manual": float, "Automated": float, "N": float})
        return df
    return pd.DataFrame(columns=["Samples", "Manual", "Automated", "N"])


def load_classified_questions():
    """Load classified questions"""
    if os.path.exists(CLASSIFIED_FILE) and os.path.getsize(CLASSIFIED_FILE) > 0:
        return pd.read_csv(CLASSIFIED_FILE)
    return pd.DataFrame(columns=['id', 'question', 'predicted_status', 'confidence', 'timestamp', 'topic'])


def save_classified_questions(results, topic):
    """Save the classified questions to a CSV file"""
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
    file_exists = os.path.isfile(CLASSIFIED_FILE) and os.path.getsize(CLASSIFIED_FILE) > 0

    if file_exists:
        existing_df = pd.read_csv(CLASSIFIED_FILE)
        next_id = existing_df['id'].max() + 1 if not existing_df.empty else 1
    else:
        next_id = 1

    df.insert(0, 'id', range(next_id, next_id + len(df)))
    df.to_csv(CLASSIFIED_FILE, mode='a', header=not file_exists, index=False)

    return len(data)


def get_true_labels(questions, feedback_data):
    """Get true labels for questions from feedback data"""
    fb = feedback_data.set_index('Question')['Status'].to_dict()
    true = []
    for q in questions:
        if q in fb:
            true.append(1 if fb[q] == 'Accepted' else 0)
    return true