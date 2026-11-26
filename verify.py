"""Recompute original metrics and compare live checkpoint outputs to saved probabilities."""

import argparse

import numpy as np
import pandas as pd

from inference import ROOT, attention_rollout, gradcam, load_cases, load_models, preprocess, probabilities


def verify(full_path=None):
    original = ROOT / "results" / "original"
    test = pd.read_csv(original / "test_model_comparison.csv").set_index("Model")
    split = pd.read_csv(original / "generalization_gap_analysis.csv").set_index("Model")
    calibration = pd.read_csv(original / "test_calibration_summary.csv").set_index("Model")
    images, labels, indices = load_cases(full_path)
    test_output = np.load(original / "test_predictions.npz", allow_pickle=False)
    validation = np.load(original / "validation_model_comparison_gpu.npz", allow_pickle=False)
    np.testing.assert_array_equal(labels, test_output["true_labels"][indices])

    for name, model in load_models().items():
        key = name.lower()
        probs = test_output[f"{key}_probabilities"]
        preds = probs.argmax(1)
        np.testing.assert_array_equal(preds, test_output[f"{key}_predictions"])
        truth = test_output["true_labels"]
        correct = preds == truth
        confidence = probs.max(1)
        bins = np.clip(np.digitize(confidence, np.linspace(0, 1, 16)) - 1, 0, 14)
        ece = sum(
            mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
            for i in range(15) if (mask := bins == i).any()
        )
        matrix = np.zeros((11, 11), dtype=int)
        np.add.at(matrix, (truth, preds), 1)
        saved_matrix = pd.read_csv(original / f"{name}_test_confusion_counts.csv", index_col=0)
        np.testing.assert_array_equal(matrix, saved_matrix.to_numpy())
        val_acc = (validation[f"{key}_predictions"] == validation["true_labels"]).mean()
        params = sum(p.numel() for p in model.parameters())
        np.testing.assert_allclose(
            [val_acc, correct.mean(), ece, params],
            [split.at[name, "Validation_Accuracy"], test.at[name, "Accuracy"],
             calibration.at[name, "ECE"], test.at[name, "Trainable_Parameters"]],
            atol=1e-7, rtol=0,
        )
        live = np.concatenate([
            probabilities(model, preprocess(images[start:start + 128]))
            for start in range(0, len(images), 128)
        ])
        # Compare CPU outputs with the archived GPU evaluation.
        deviation = float(np.max(np.abs(live - probs[indices])))
        matching = int(np.sum(live.argmax(1) == preds[indices]))
        assert deviation <= 0.002, f"{name}: probability deviation {deviation:.6f}"
        assert matching / len(images) >= 0.999, f"{name}: only {matching} argmax matches"
        for i in range(min(12, len(images))):
            inputs = preprocess(images[i:i + 1])
            heatmap = (gradcam(model, inputs, int(live[i].argmax()))
                       if name == "CNN" else attention_rollout(model, inputs))
            assert heatmap.shape == (64, 64) and np.isfinite(heatmap).all()
            assert heatmap.min() >= 0 and heatmap.max() <= 1
        print(
            f"{name}: {matching:,}/{len(images):,} argmax matches | validation {val_acc:.2%} "
            f"| test {correct.mean():.2%} | {params:,} weights | ECE {ece:.6f} "
            f"| max probability deviation {deviation:.3g}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Path to exact full OrganAMNIST-64 archive")
    verify(parser.parse_args().dataset)
