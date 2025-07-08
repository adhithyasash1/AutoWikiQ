import streamlit as st
from data_handler import load_feedback_data, load_acceptance_history, load_classified_questions

def initialize_session_state():
    """Initialize all session state variables"""
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
        st.session_state.current_mode = "Generate"
    if 'run_offset' not in st.session_state:
        st.session_state.run_offset = 0