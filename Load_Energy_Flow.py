"""
Load_Energy_Flow.py
-------------------
Download/load the EnergyFlow quark-vs-gluon dataset and save it locally.

Expected dataset layout:
    X[:, :, 0] = particle pT
    X[:, :, 1] = particle rapidity y
    X[:, :, 2] = particle phi
    X[:, :, 3] = particle PDG ID
    y = quark/gluon label (quark=1, gluon=0)

Run from anywhere:
    python setup/Load_Energy_Flow.py
"""

from pathlib import Path
import json
import numpy as np
import energyflow as ef

# Project root = directory containing this setup/ folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Start with 5000 jets, matching the dataset you already downloaded.
NUM_JETS = 5000

print("=" * 60)
print("EnergyFlow Quark/Gluon Dataset Loader")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Saving to:    {DATA_DIR}")
print()

print("Loading EnergyFlow quark/gluon dataset...")
X, y = ef.qg_jets.load(
    num_data=NUM_JETS,
    pad=True,
    ncol=4,
    generator="pythia",
    with_bc=False,
)

print("\nDataset information")
print("-" * 60)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of jets: {len(X)}")
print(f"Maximum particles: {X.shape[1]}")
print(f"Features per particle: {X.shape[2]}")

labels, counts = np.unique(y, return_counts=True)
print("\nLabels:")
for label, count in zip(labels, counts):
    name = "quark" if int(label) == 1 else "gluon" if int(label) == 0 else "unknown"
    print(f"  {int(label)} = {name:6s}: {count}")

print("\nParticle features:")
features = ["pt", "rapidity", "phi", "pid"]
for i, name in enumerate(features):
    print(f"  column {i}: {name}")

# Save arrays.
x_path = DATA_DIR / "qg_X.npy"
y_path = DATA_DIR / "qg_y.npy"
np.save(x_path, X)
np.save(y_path, y)

# Save a small metadata file so the analysis is reproducible.
metadata = {
    "dataset": "EnergyFlow qg_jets",
    "generator": "pythia",
    "with_bc": False,
    "num_jets": int(len(X)),
    "shape_X": list(X.shape),
    "shape_y": list(y.shape),
    "features": {
        "0": "pt",
        "1": "rapidity",
        "2": "phi",
        "3": "pid",
    },
    "labels": {
        "0": "gluon",
        "1": "quark",
    },
    "selection": {
        "sqrt_s_TeV": 14,
        "jet_pt_GeV": [500, 550],
        "abs_rapidity_max": 1.7,
        "jet_R": 0.4,
    },
}

with open(DATA_DIR / "qg_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved:")
print(f"  {x_path}")
print(f"  {y_path}")
print(f"  {DATA_DIR / 'qg_metadata.json'}")
print("\nLoader finished successfully.")
