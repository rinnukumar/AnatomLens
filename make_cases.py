"""Build the offline explorer subset from the exact MedMNIST release."""

import argparse

import numpy as np
import pandas as pd

from inference import NAMES, ROOT, load_cases


def main(path):
    images, labels, _ = load_cases(path)
    original = ROOT / "results" / "original"
    examples = pd.read_csv(original / "selected_test_examples.csv")
    chosen = list(examples["Test_Index"])
    with np.load(original / "test_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(labels, saved["true_labels"])
        cnn_ok = saved["cnn_predictions"] == labels
        vit_ok = saved["vit_predictions"] == labels
    rng = np.random.default_rng(42)
    for label in range(len(NAMES)):
        for cnn, vit in ((True, True), (True, False), (False, True), (False, False)):
            matches = np.flatnonzero((labels == label) & (cnn_ok == cnn) & (vit_ok == vit))
            chosen.extend(rng.choice(matches, size=min(4, len(matches)), replace=False))
    indices = np.array(list(dict.fromkeys(chosen)))
    np.savez_compressed(
        ROOT / "data" / "explorer_cases.npz",
        images=images[indices], labels=labels[indices], indices=indices,
    )
    print(f"Packaged {len(indices)} actual test cases across {len(np.unique(labels[indices]))} organs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    main(parser.parse_args().dataset)
