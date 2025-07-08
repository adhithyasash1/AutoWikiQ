import pandas as pd
import numpy as np
import os
from config import FEEDBACK_CSV, ACCEPTANCE_HISTORY_CSV, CLASSIFIED_QUESTIONS_CSV


def load_feedback_data():
    """Load existing feedback data if available"""
    if os.path.exists(FEEDBACK_CSV) and os.path.getsize(FEEDBACK_CSV) > 0:
        return pd.read_csv(FEEDBACK_CSV)
    return pd.DataFrame(columns=['Question', 'Status', 'Timestamp'])


def load_acceptance_history():
    """Load acceptance rate history"""
    if os.path.exists(ACCEPTANCE_HISTORY_CSV) and os.path.getsize(ACCEPTANCE_HISTORY_CSV) > 0:
        df = pd.read_csv(ACCEPTANCE_HISTORY_CSV)
        # Ensure "N" column exists
        if "N" not in df.columns:
            df["N"] = np.nan
        # Cast dtypes for consistency
        df = df.astype({"Samples": float, "Manual": float, "Automated": float, "N": float})
        return df
    return pd.DataFrame(columns=["Samples", "Manual", "Automated", "N"])


def load_classified_questions():
    """Load classified questions"""
    if os.path.exists(CLASSIFIED_QUESTIONS_CSV) and os.path.getsize(CLASSIFIED_QUESTIONS_CSV) > 0:
        return pd.read_csv(CLASSIFIED_QUESTIONS_CSV)
    return pd.DataFrame(columns=['id', 'question', 'predicted_status', 'confidence', 'timestamp', 'topic'])


def save_classified_questions(results, topic):
    """Save the classified questions to a CSV file"""
    from datetime import datetime

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
    file_exists = os.path.isfile(CLASSIFIED_QUESTIONS_CSV) and os.path.getsize(CLASSIFIED_QUESTIONS_CSV) > 0

    # Get the next ID value
    if file_exists:
        existing_df = pd.read_csv(CLASSIFIED_QUESTIONS_CSV)
        next_id = existing_df['id'].max() + 1 if not existing_df.empty else 1
    else:
        next_id = 1

    # Add ID column
    df.insert(0, 'id', range(next_id, next_id + len(df)))

    # Append to existing file or create new one
    df.to_csv(CLASSIFIED_QUESTIONS_CSV, mode='a', header=not file_exists, index=False)

    return len(data)


def get_true_labels(questions, feedback_data):
    """
    Given a list of question strings, look up their human 'Accepted'/'Rejected'
    status in feedback_data and return [0|1] for each.
    If a question is not found, we skip it.
    """
    fb = feedback_data.set_index('Question')['Status'].to_dict()
    true = []
    for q in questions:
        if q in fb:
            true.append(1 if fb[q] == 'Accepted' else 0)
    return true