**AutoWikiQ: Adaptive Wikipedia Question‑Generation & Classification**

---

## Overview

AutoWikiQ is a Python application built with Streamlit that automates the generation and evaluation of research-style questions from Wikipedia articles. It integrates human feedback with machine learning to progressively reduce manual review effort and maximize question quality.

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
