import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from question_generator import fetch_wikipedia_content, generate_questions_from_text
from classifier import classify_questions
from data_handler import save_classified_questions, get_true_labels


def run_pipeline_mode(topic, N, threshold, session_state):
    """Execute the 6-step pipeline for generating and classifying questions"""
    prev_autos = session_state.acceptance_history["Automated"].dropna()
    last_auto = prev_autos.iloc[-1] if not prev_autos.empty else None

    with st.spinner("Running the 6-step pipeline…"):
        for attempt in (False, True):
            all_res = []
            M = 0

            # Step 0-1: fetch & initial generate
            text = fetch_wikipedia_content(topic)
            if not text:
                st.error("Topic not found on Wikipedia.")
                st.stop()

            initial_qs = generate_questions_from_text(text, N, variant=attempt)
            initial_res = classify_questions(initial_qs, session_state.vectorizer, session_state.model)
            all_res += initial_res
            M += len(initial_res)

            # Step 2-4: top-up until we have N accepted
            accepted = [r for r in all_res if r["Confidence"] >= threshold]
            while len(accepted) < N:
                to_gen = N - len(accepted)
                extra_qs = generate_questions_from_text(text, to_gen, variant=attempt)
                extra_res = classify_questions(extra_qs, session_state.vectorizer, session_state.model)
                all_res += extra_res
                M += len(extra_res)
                accepted = [r for r in all_res if r["Confidence"] >= threshold]

            # Step 5: select top N
            final_accepted = sorted(accepted, key=lambda r: r["Confidence"], reverse=True)[:N]

            # Persist results
            save_classified_questions(all_res, topic)
            auto_rate = N / M * 100

            # If first attempt and auto_rate unchanged, retry with variant
            if attempt is False and last_auto is not None and auto_rate == last_auto:
                st.warning("Automated acceptance rate unchanged; retrying with a slightly tweaked prompt…")
                continue

            break

        # Show final results
        st.success(f"Final {len(final_accepted)} accepted questions out of {M} generated")
        for r in final_accepted:
            st.write(f"- {r['Question']}  (conf={r['Confidence']:.2f})")

        # Compute manual acceptance on this run's slice
        fb = session_state.feedback_data.sort_values("Timestamp").reset_index(drop=True)
        start = session_state.run_offset
        end = start + M
        segment = fb.iloc[start:end]
        manual_seg = segment["Status"].eq("Accepted").mean() * 100 if len(segment) > 0 else 0
        session_state.run_offset = end

        st.info(f"Automated acceptance rate: **{auto_rate:.1f}%**    "
                f"(Manual over this run: {manual_seg:.1f}%)")

        # Update history
        new_run = pd.DataFrame({
            "Samples": [end],
            "Manual": [manual_seg],
            "Automated": [auto_rate],
            "N": [N]
        }).dropna(subset=["Automated"])

        hist = pd.concat([session_state.acceptance_history, new_run], ignore_index=True)
        hist = hist.drop_duplicates(keep="last")
        session_state.acceptance_history = hist
        hist.to_csv("acceptance_history.csv", index=False)

        # Batch accuracy
        true_labels = get_true_labels([r["Question"] for r in final_accepted], session_state.feedback_data)
        if true_labels:
            acc = accuracy_score(true_labels, [1] * len(true_labels))
            st.info(f"Classifier accuracy on overlap: **{acc * 100:.1f}%**")
        else:
            st.warning("No overlap with manual labels → cannot compute batch accuracy.")