import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_analytics_mode(session_state):
    """Plot manual vs automated acceptance rate analytics"""
    hist = session_state.acceptance_history.copy()

    if hist.empty:
        st.warning("No pipeline runs yet—run Test Classification first.")
    else:
        # Prepend (0,0) baseline if missing
        if hist["Samples"].min() > 0:
            zero = pd.DataFrame({
                "Samples": [0],
                "Manual": [0.0],
                "Automated": [0.0],
                "N": [0]
            })
            hist = pd.concat([zero, hist], ignore_index=True)

        hist = hist.sort_values("Samples").reset_index(drop=True)

        # Per-run x and y values
        x = hist["Samples"]
        y_man = hist["Manual"]
        y_auto = hist["Automated"]

        # Compute cumulative efficiency
        cum_accepted = hist["N"].cumsum()
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

        # Annotate per-run and cumulative
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