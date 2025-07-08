import streamlit as st
from session_state import initialize_session_state
from ui_modes import mode_generate_feedback, mode_train_classifier, mode_test_classification, mode_view_analytics

# Streamlit page setup
st.set_page_config(page_title="Wiki Question Generator", layout="centered")
st.title("Wikipedia Question Generator with Feedback")

# Initialize session state
initialize_session_state()

# Sidebar + modes
st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Select Mode",
                        ["Generate & Feedback", "Train Classifier", "Test Classification", "View Analytics"])
st.session_state.current_mode = mode

# Users can tune how the automated loop decides "accepted"
threshold = st.sidebar.slider(
    "Auto-Accept Confidence Threshold",
    min_value=0.0, max_value=1.0, value=0.7,
    help="When auto-selecting questions, only those ≥ this confidence count."
)

# Mode routing
if mode == "Generate & Feedback":
    mode_generate_feedback()
elif mode == "Train Classifier":
    mode_train_classifier()
elif mode == "Test Classification":
    mode_test_classification(threshold)
elif mode == "View Analytics":
    mode_view_analytics()
else:
    st.warning("No feedback data available. Please generate questions and provide feedback in Step 1 first.")