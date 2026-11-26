"""AnatomLens Streamlit research demonstration."""

from pathlib import Path
from time import perf_counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from inference import NAMES, attention_rollout, gradcam, load_cases, load_models, overlay, preprocess, probabilities

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "results" / "original"
CONTROLLED = ROOT / "results" / "controlled"
COLORS = ["#087e76", "#6549a2"]

st.set_page_config(page_title="AnatomLens", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1210px; padding-top: 3rem;}
div[data-testid="stMetric"] {background: white; border: 1px solid #dce8ec; border-radius: 12px; padding: 1rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: white;}
.stTabs [role="tab"] {font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def models():
    return load_models()


@st.cache_data
def cases():
    return load_cases()


@st.cache_data
def prediction(position):
    images, _, _ = cases()
    image = images[position]
    inputs = preprocess(image[None])
    cnn, vit = models().values()
    cnn_prob = probabilities(cnn, inputs)[0]
    vit_prob = probabilities(vit, inputs)[0]
    return (
        cnn_prob, vit_prob,
        overlay(image, gradcam(cnn, inputs, int(cnn_prob.argmax()))),
        overlay(image, attention_rollout(vit, inputs)),
    )


def original_scores():
    test = pd.read_csv(ORIGINAL / "test_model_comparison.csv").set_index("Model")
    split = pd.read_csv(ORIGINAL / "generalization_gap_analysis.csv").set_index("Model")
    calibration = pd.read_csv(ORIGINAL / "test_calibration_summary.csv").set_index("Model")
    return test, split, calibration


def overview():
    st.subheader("Does a larger model make more reliable predictions?")
    st.write(
        "An IIT Madras dual-degree project examining organ classification on 64 × 64 axial CT images. "
        "Compare two trained models on the same held-out images, then inspect where their errors and confidence differ."
    )
    test, split, calibration = original_scores()
    for name, column in zip(("CNN", "ViT"), st.columns(2)):
        with column.container(border=True):
            st.markdown(f"### {name}")
            accuracy, validation, size = st.columns(3)
            accuracy.metric("Test accuracy", f"{test.at[name, 'Accuracy']:.2%}")
            validation.metric("Validation accuracy", f"{split.at[name, 'Validation_Accuracy']:.2%}")
            size.metric("Trainable parameters", f"{int(test.at[name, 'Trainable_Parameters']):,}")
            st.caption(f"Test ECE · {calibration.at[name, 'ECE']:.6f} · 15 equal-width bins")
    st.info("The original training protocols differed. The test gap motivates a controlled follow-up.")
    st.markdown("#### Research scope")
    st.write(
        "OrganAMNIST is a benchmark of grayscale organ crops. These models classify benchmark images into "
        "11 organ classes. External validation, clinical workflow testing, and patient-level safety assessment "
        "are outside the scope of these experiments. Use this demo for research and education."
    )
    st.caption("Research by Akriti Kumar · BS-MS Dual Degree Biotechnology · IIT Madras")


def explorer():
    st.subheader("Inspect a held-out CT image")
    try:
        images, labels, indices = cases()
        models()
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Explorer setup failed. See the README. {exc}")
        return
    with np.load(ORIGINAL / "test_predictions.npz", allow_pickle=False) as saved:
        cnn_ok = saved["cnn_predictions"][indices] == labels
        vit_ok = saved["vit_predictions"][indices] == labels
    outcomes = {
        "All cases": np.ones(len(labels), dtype=bool),
        "Both correct": cnn_ok & vit_ok,
        "CNN only correct": cnn_ok & ~vit_ok,
        "ViT only correct": ~cnn_ok & vit_ok,
        "Both incorrect": ~cnn_ok & ~vit_ok,
    }
    left, right = st.columns(2)
    organ = left.selectbox("Ground-truth organ filter", ["All organs", *NAMES])
    outcome = right.selectbox("Saved test outcome filter", outcomes)
    mask = outcomes[outcome]
    if organ != "All organs":
        mask = mask & (labels == NAMES.index(organ))
    choices = np.flatnonzero(mask)
    if not len(choices):
        st.info("No cases match these filters. Choose a different organ or outcome.")
        return
    position = st.selectbox(
        "Test case", choices.tolist(), format_func=lambda i: f"Test index {indices[i]} · {NAMES[labels[i]]}"
    )
    truth = int(labels[position])
    st.caption(f"{len(choices):,} matching cases · benchmark test index {indices[position]:,}")
    cnn_prob, vit_prob, cnn_map, vit_map = prediction(position)
    base, cnn_col, vit_col = st.columns(3)
    with base.container(border=True):
        st.markdown("#### Ground truth")
        st.write(NAMES[truth])
        st.image(images[position], clamp=True, width=220)
    for column, name, values, heatmap, method in (
        (cnn_col, "CNN", cnn_prob, cnn_map, "Grad-CAM"),
        (vit_col, "ViT", vit_prob, vit_map, "Attention rollout"),
    ):
        with column.container(border=True):
            predicted = int(values.argmax())
            st.markdown(f"#### {name}")
            st.write(f"**{NAMES[predicted]}** · {values[predicted]:.1%} confidence")
            st.image(heatmap, width=220)
            st.caption(f"{method} · {'Correct prediction' if predicted == truth else 'Incorrect prediction'}")
    st.caption("Grad-CAM targets the predicted class. Rollout traces attention from the class token. Both maps use per-image normalization and have limited explanatory scope.")
    distributions = pd.DataFrame({"Organ": NAMES, "CNN": cnn_prob, "ViT": vit_prob})
    chart = alt.Chart(distributions.melt("Organ", var_name="Model", value_name="Probability"))
    st.altair_chart(chart.mark_bar().encode(
        y=alt.Y("Organ:N", sort=list(NAMES)),
        yOffset=alt.YOffset("Model:N"),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["CNN", "ViT"], range=COLORS)),
        tooltip=["Organ:N", "Model:N", alt.Tooltip("Probability:Q", format=".2%")],
    ).properties(height=360))


def diagnostics():
    st.subheader("Original test-set diagnostics")
    view = st.segmented_control(
        "Analysis", ["Class errors", "Calibration", "Efficiency"], default="Class errors"
    )
    test, _, calibration = original_scores()
    if view == "Class errors":
        st.caption("Rows are true organs. Columns are predictions. Color shows the share of each true class.")
        for name, col in zip(("CNN", "ViT"), st.columns(2)):
            col.image(str(ROOT / "figures" / f"{name}_test_confusion_normalized.png"), caption=name)
        st.dataframe(pd.read_csv(ORIGINAL / "test_classwise_comparison.csv"), hide_index=True)
    elif view == "Calibration":
        left, right = st.columns([1.05, 1])
        left.image(str(ROOT / "figures" / "test_reliability_diagram.png"))
        right.dataframe(calibration[["ECE", "Mean_Confidence", "Incorrect_Confidence", "NLL", "Brier_Score"]])
        right.caption("ECE is the sample-weighted absolute confidence–accuracy gap across 15 equal-width bins.")
        for name, col in zip(("CNN", "ViT"), st.columns(2)):
            col.image(str(ROOT / "figures" / f"{name}_test_confidence.png"))
    elif view == "Efficiency":
        st.write(f"The original ViT has {test.at['ViT', 'Trainable_Parameters'] / test.at['CNN', 'Trainable_Parameters']:.2f}× as many trainable parameters as the CNN.")
        st.bar_chart(test["Trainable_Parameters"], x_label="Model", y_label="Trainable parameters")
        if st.button("Measure CPU inference"):
            try:
                with st.spinner("Timing both checkpoints…"):
                    timing = benchmark()
                st.dataframe(timing, hide_index=True)
                st.caption("CPU, two PyTorch threads, batch size 1. Median of 30 forwards after 5 warm-ups, excluding loading and explanations.")
            except (OSError, RuntimeError) as exc:
                st.error(f"Benchmark unavailable. {exc}")


def benchmark():
    images, _, _ = cases()
    inputs = preprocess(images[:1])
    rows = []
    with torch.inference_mode():
        for name, model in models().items():
            for _ in range(5):
                model(inputs)
            times = []
            for _ in range(30):
                start = perf_counter()
                model(inputs)
                times.append((perf_counter() - start) * 1000)
            rows.append({"Model": name, "Median forward (ms)": round(float(np.median(times)), 3)})
    return pd.DataFrame(rows)


def controlled():
    st.subheader("Controlled comparison · progress")
    st.warning("Five of six planned runs completed. ViT seed 2026 has 8 of 20 epochs in the supplied history and no completed summary.")
    st.write("The matched protocol planned 20 epochs for each model at seeds 42, 123 and 2026, using AdamW, a batch size of 128, and best-checkpoint selection by validation macro-F1.")
    table = pd.read_csv(CONTROLLED / "controlled_validation_summary.csv")
    columns = ["Model", "Seed", "Epochs_Completed", "Best_Epoch", "Val_Accuracy", "Val_Macro_F1"]
    st.dataframe(
        table[columns], hide_index=True,
        column_config={"Val_Accuracy": st.column_config.NumberColumn(format="%.4f"),
                       "Val_Macro_F1": st.column_config.NumberColumn(format="%.4f")},
    )
    st.caption("Completed runs show validation metrics only. The archive contains no finalized controlled test comparison.")


st.caption("AKRITI KUMAR  /  IIT MADRAS")
st.title("AnatomLens")
st.markdown("### AI Model Reliability Explorer")
page = st.radio("View", ["Overview", "CT explorer", "Test diagnostics", "Controlled runs"],
                horizontal=True, label_visibility="collapsed")
{"Overview": overview, "CT explorer": explorer, "Test diagnostics": diagnostics,
 "Controlled runs": controlled}[page]()
