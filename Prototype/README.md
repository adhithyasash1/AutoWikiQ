**AutoWikiQ: Adaptive Wikipedia Question‑Generation & Classification**

---

## Overview

AutoWikiQ is a full-stack Python application built with Streamlit that automates the generation and evaluation of research-style questions from Wikipedia articles. It integrates human feedback with machine learning to progressively reduce manual review effort and maximize question quality.

## Key Features

1. **Fetch & Generate**

   * Pulls text from any English Wikipedia page (up to a configurable character limit) using `wikipediaapi`.
   * Generates N candidate questions via Google Gemini LLM, with prompt variants to avoid duplication.

2. **Human-in-the-Loop Feedback**

   * Presents generated questions in a Streamlit UI for manual "Accept" or "Reject" labeling.
   * Stores labeled data (`questions_feedback.csv`) for training.

3. **Classifier Training**

   * Trains a TF-IDF + Random Forest model on human labels once at least 10 samples are available.
   * Saves the trained vectorizer and model to disk (`question_classifier.pkl`).

4. **Automated 6‑Step Pipeline**

   * Generates an initial batch of questions.
   * Uses the classifier to filter for high-confidence "Accepted" questions.
   * Iteratively top‑up generation until the desired N accepted questions are obtained.
   * Persists all classification results (`classified_questions.csv`) and logs efficiency metrics (`acceptance_history.csv`).
   * Retries with prompt variants if automated acceptance rate stagnates.

5. **Analytics & Visualization**

   * Compares manual vs. automated acceptance rates per run and cumulatively.
   * Plots efficiency curves with Matplotlib in a Streamlit dashboard.

## Purpose & Outcomes

* **Purpose of Training the Classifier**
  To learn from human judgments which questions are high-quality, enabling the system to automatically filter future LLM outputs without exhaustive manual review.

* **End Achievements**

  * A reproducible pipeline that generates N high-confidence, research-style questions on any Wikipedia topic in seconds.
  * A trained model capable of instantly predicting question acceptance with confidence scores.
  * Quantitative analytics showing reduction in human effort (e.g., calls-per-accepted-question) and a record of automatic vs. manual acceptance rates.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/AutoWikiQ.git
   cd AutoWikiQ
   ```

2. **Set up environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API Key**

   ```bash
   export GEMINI_API_KEY="your_gemini_key_here"
   ```

## Usage

```bash
streamlit run app.py
```

* **Mode 1**: Generate & Feedback (label questions)
* **Mode 2**: Train Classifier (needs ≥ 10 labeled samples)
* **Mode 3**: Test Classification (runs the automated pipeline)
* **Mode 4**: View Analytics (plots acceptance metrics)

## File Structure

```
├── app.py                      # Main Streamlit script
├── questions_feedback.csv     # Human-labeled questions & status
├── acceptance_history.csv      # Pipeline run metrics (M, N, rates)
├── classified_questions.csv    # All auto-classified questions
├── question_classifier.pkl     # Pickled TF-IDF & Random Forest model
├── requirements.txt            # Python dependencies
```