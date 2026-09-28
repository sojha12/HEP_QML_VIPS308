"""
Jet_analysis.py
---------------
Generate physics observables, plots, and ML-ready numerical tables from
the locally saved EnergyFlow quark/gluon dataset.

Run:
    python setup/Jet_analysis.py

Outputs:
    data/figures/*.png
    data/analysis/qg_event_observables.csv
    data/analysis/qg_summary_by_class.csv
    data/analysis/qg_top5_values.npz

The analysis uses:
    X[:, :, 0] = particle pT
    X[:, :, 1] = particle rapidity
    X[:, :, 2] = particle phi
    X[:, :, 3] = particle PID

Important:
    Zero-padded particles are removed with pt > 0.

The calculations below are intended as a reproducible analysis/recreation
pipeline. They are not claims about the numerical results of the paper.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "data" / "figures"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_DIR / "qg_X.npy"
Y_PATH = DATA_DIR / "qg_y.npy"

if not X_PATH.exists() or not Y_PATH.exists():
    raise FileNotFoundError(
        "\nCould not find the EnergyFlow arrays.\n"
        "Run first:\n\n"
        "    python setup/Load_Energy_Flow.py\n"
    )

# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

print("=" * 70)
print("EnergyFlow Quark/Gluon Physics Analysis")
print("=" * 70)

X = np.load(X_PATH)
y = np.load(Y_PATH)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

pt = X[:, :, 0].astype(float)
rapidity = X[:, :, 1].astype(float)
phi = X[:, :, 2].astype(float)
pid = X[:, :, 3]

# Zero-padded entries are not physical constituents.
mask = pt > 0

# ---------------------------------------------------------------------
# Basic event quantities
# ---------------------------------------------------------------------

multiplicity = mask.sum(axis=1)

jet_pt = np.sum(np.where(mask, pt, 0.0), axis=1)

# Four-vector sums, treating constituents as massless.
px = np.sum(np.where(mask, pt * np.cos(phi), 0.0), axis=1)
py = np.sum(np.where(mask, pt * np.sin(phi), 0.0), axis=1)
pz = np.sum(np.where(mask, pt * np.sinh(rapidity), 0.0), axis=1)
energy = np.sum(np.where(mask, pt * np.cosh(rapidity), 0.0), axis=1)

jet_mass_sq = np.maximum(energy**2 - px**2 - py**2 - pz**2, 0.0)
jet_mass = np.sqrt(jet_mass_sq)

jet_axis_y = 0.5 * np.log(
    np.maximum(energy + pz, 1e-12) /
    np.maximum(energy - pz, 1e-12)
)
jet_axis_phi = np.arctan2(py, px)

# ---------------------------------------------------------------------
# Momentum fractions z_i
# ---------------------------------------------------------------------

z = np.divide(
    pt,
    jet_pt[:, None],
    out=np.zeros_like(pt),
    where=jet_pt[:, None] > 0,
)

# Sort constituents by pT so the analysis is independent of the input
# ordering.
sort_index = np.argsort(-pt, axis=1)
pt_sorted = np.take_along_axis(pt, sort_index, axis=1)
z_sorted = np.take_along_axis(z, sort_index, axis=1)
y_sorted = np.take_along_axis(rapidity, sort_index, axis=1)
phi_sorted = np.take_along_axis(phi, sort_index, axis=1)
pid_sorted = np.take_along_axis(pid, sort_index, axis=1)

mask_sorted = np.take_along_axis(mask, sort_index, axis=1)

# ---------------------------------------------------------------------
# Angular distance from reconstructed jet axis
# ---------------------------------------------------------------------

dy = y_sorted - jet_axis_y[:, None]
dphi = (phi_sorted - jet_axis_phi[:, None] + np.pi) % (2 * np.pi) - np.pi
dr = np.sqrt(dy**2 + dphi**2)
dr = np.where(mask_sorted, dr, 0.0)

# ---------------------------------------------------------------------
# Physics observables
# ---------------------------------------------------------------------

# pT-weighted jet girth / width:
#     g = sum_i z_i * DeltaR_i
girth = np.sum(z_sorted * dr, axis=1)

# Generalized angularity-like observables:
#     lambda_beta = sum_i z_i * (DeltaR_i)^beta
angularity_05 = np.sum(z_sorted * dr**0.5, axis=1)
angularity_1 = np.sum(z_sorted * dr, axis=1)
angularity_2 = np.sum(z_sorted * dr**2, axis=1)

# Radial second moment:
radial_second_moment = angularity_2

# ---------------------------------------------------------------------
# Top-N truncation information
# ---------------------------------------------------------------------

N_TOP = 5

z_top5 = z_sorted[:, :N_TOP]
pt_top5 = pt_sorted[:, :N_TOP]

retained_top5 = np.sum(z_top5, axis=1)
lost_top5 = 1.0 - retained_top5

# Cumulative retained momentum for N=1..10.
N_CUM = min(10, X.shape[1])
cumulative_retained = np.cumsum(z_sorted[:, :N_CUM], axis=1)
# ---------------------------------------------------------------------
# Useful particle-level distributions
# ---------------------------------------------------------------------

real_z = z_sorted[mask_sorted]
real_dr = dr[mask_sorted]
real_pt = pt_sorted[mask_sorted]

# ---------------------------------------------------------------------
# Save ML-ready event table
# ---------------------------------------------------------------------

event_data = {
    "label": y.astype(int),
    "jet_pt_GeV": jet_pt,
    "jet_mass_GeV": jet_mass,
    "multiplicity": multiplicity,
    "jet_axis_y": jet_axis_y,
    "jet_axis_phi": jet_axis_phi,
    "girth": girth,
    "angularity_beta_0p5": angularity_05,
    "angularity_beta_1": angularity_1,
    "angularity_beta_2": angularity_2,
    "retained_pt_fraction_top5": retained_top5,
    "lost_pt_fraction_top5": lost_top5,
}

for i in range(N_TOP):
    event_data[f"z_{i+1}"] = z_top5[:, i]
    event_data[f"pt_{i+1}_GeV"] = pt_top5[:, i]

for i in range(N_CUM):
    event_data[f"retained_pt_fraction_top_{i+1}"] = cumulative_retained[:, i]

df = pd.DataFrame(event_data)

csv_path = ANALYSIS_DIR / "qg_event_observables.csv"
df.to_csv(csv_path, index=False)

# Compact NPZ for later ML/QML scripts.
np.savez(
    ANALYSIS_DIR / "qg_top5_values.npz",
    y=y,
    z_top5=z_top5,
    pt_top5=pt_top5,
    cumulative_retained=cumulative_retained,
    multiplicity=multiplicity,
    jet_pt=jet_pt,
    jet_mass=jet_mass,
    girth=girth,
    angularity_beta_0p5=angularity_05,
    angularity_beta_1=angularity_1,
    angularity_beta_2=angularity_2,
    lost_top5=lost_top5,
)

# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

rows = []

for label, name in [(0, "gluon"), (1, "quark")]:
    sel = y == label

    rows.append({
        "class": name,
        "n_jets": int(sel.sum()),
        "mean_multiplicity": multiplicity[sel].mean(),
        "std_multiplicity": multiplicity[sel].std(),
        "mean_jet_pt_GeV": jet_pt[sel].mean(),
        "mean_jet_mass_GeV": jet_mass[sel].mean(),
        "mean_girth": girth[sel].mean(),
        "mean_angularity_beta_0p5": angularity_05[sel].mean(),
        "mean_angularity_beta_1": angularity_1[sel].mean(),
        "mean_angularity_beta_2": angularity_2[sel].mean(),
        "mean_retained_top5": retained_top5[sel].mean(),
        "mean_lost_top5": lost_top5[sel].mean(),
    })

summary = pd.DataFrame(rows)
summary.to_csv(ANALYSIS_DIR / "qg_summary_by_class.csv", index=False)

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def class_hist(values, xlabel, filename, bins=50, log_y=False):
    plt.figure(figsize=(7.5, 5.5))

    plt.hist(
        values[y == 0],
        bins=bins,
        density=True,
        alpha=0.55,
        label="Gluon",
    )

    plt.hist(
        values[y == 1],
        bins=bins,
        density=True,
        alpha=0.55,
        label="Quark",
    )

    plt.xlabel(xlabel)
    plt.ylabel("Normalized density")
    plt.legend()
    plt.tight_layout()

    if log_y:
        plt.yscale("log")

    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------
# FIGURE 01 — constituent multiplicity
# ---------------------------------------------------------------------

class_hist(
    multiplicity,
    "Number of jet constituents",
    "01_constituent_multiplicity.png",
    bins=np.arange(multiplicity.max() + 2) - 0.5,
)

# ---------------------------------------------------------------------
# FIGURE 02 — jet pT
# ---------------------------------------------------------------------

class_hist(
    jet_pt,
    r"Jet $p_T$ [GeV]",
    "02_jet_pt.png",
    bins=40,
)

# ---------------------------------------------------------------------
# FIGURE 03 — jet mass
# ---------------------------------------------------------------------

class_hist(
    jet_mass,
    r"Jet mass [GeV]",
    "03_jet_mass.png",
    bins=50,
)

# ---------------------------------------------------------------------
# FIGURE 04 — jet girth / width
# ---------------------------------------------------------------------

class_hist(
    girth,
    r"Jet girth  $\sum_i z_i \Delta R_i$",
    "04_jet_girth.png",
    bins=50,
)

# ---------------------------------------------------------------------
# FIGURE 05 — angularity beta = 0.5
# ---------------------------------------------------------------------

class_hist(
    angularity_05,
    r"Angular observable  $\sum_i z_i(\Delta R_i)^{0.5}$",
    "05_angularity_beta_0p5.png",
    bins=50,
)

# ---------------------------------------------------------------------
# FIGURE 06 — angularity beta = 1
# ---------------------------------------------------------------------

class_hist(
    angularity_1,
    r"Angular observable  $\sum_i z_i\Delta R_i$",
    "06_angularity_beta_1.png",
    bins=50,
)

# ---------------------------------------------------------------------
# FIGURE 07 — angularity beta = 2
# ---------------------------------------------------------------------

class_hist(
    angularity_2,
    r"Angular observable  $\sum_i z_i(\Delta R_i)^2$",
    "07_angularity_beta_2.png",
    bins=50,
)

# ---------------------------------------------------------------------
# FIGURE 08 — top-5 retained momentum
# ---------------------------------------------------------------------

class_hist(
    retained_top5,
    r"Momentum fraction retained by 5 hardest constituents",
    "08_top5_retained_fraction.png",
    bins=50,
)

# ---------------------------------------------------------------------
# FIGURE 09 — top-5 lost momentum
# ---------------------------------------------------------------------

class_hist(
    lost_top5,
    r"Momentum fraction outside 5 hardest constituents",
    "09_top5_lost_fraction.png",
    bins=50,
)

# ---------------------------------------------------------------------
# FIGURE 10 — constituent z distribution
# ---------------------------------------------------------------------

plt.figure(figsize=(7.5, 5.5))
plt.hist(real_z, bins=np.logspace(-5, 0, 70), density=True)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Constituent momentum fraction $z=p_T^i/p_T^{jet}$")
plt.ylabel("Normalized density")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "10_constituent_z_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ---------------------------------------------------------------------
# FIGURE 11 — constituent radial distribution
# ---------------------------------------------------------------------

plt.figure(figsize=(7.5, 5.5))
plt.hist(real_dr, bins=50, density=True)
plt.xlabel(r"Constituent distance $\Delta R$ from jet axis")
plt.ylabel("Normalized density")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "11_constituent_deltaR.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ---------------------------------------------------------------------
# FIGURE 12 — cumulative retained momentum vs N
# ---------------------------------------------------------------------

N_values = np.arange(1, N_CUM + 1)

plt.figure(figsize=(7.5, 5.5))

for label, name in [(0, "Gluon"), (1, "Quark")]:
    sel = y == label
    mean_curve = cumulative_retained[sel].mean(axis=0)
    std_curve = cumulative_retained[sel].std(axis=0)

    plt.plot(N_values, mean_curve, marker="o", label=name)
    plt.fill_between(
        N_values,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15,
    )

plt.xlabel("Number of hardest constituents retained")
plt.ylabel(r"Mean retained momentum fraction $\sum_{i=1}^{N} z_i$")
plt.ylim(0, 1.05)
plt.xticks(N_values)
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "12_cumulative_retained_momentum.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ---------------------------------------------------------------------
# FIGURE 13 — top 5 z values
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5.5))

for label, name in [(0, "Gluon"), (1, "Quark")]:
    sel = y == label
    means = z_top5[sel].mean(axis=0)
    stds = z_top5[sel].std(axis=0)

    xvals = np.arange(1, N_TOP + 1)
    plt.errorbar(xvals, means, yerr=stds, marker="o", capsize=3, label=name)

plt.xlabel("Constituent rank by $p_T$")
plt.ylabel(r"$z_i$")
plt.xticks(range(1, N_TOP + 1))
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "13_top5_z_values.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ---------------------------------------------------------------------
# Print important numerical values
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)

print("\nClass counts:")
print(summary[["class", "n_jets"]].to_string(index=False))

print("\nMean observables:")
print(
    summary[
        [
            "class",
            "mean_multiplicity",
            "mean_jet_pt_GeV",
            "mean_jet_mass_GeV",
            "mean_girth",
            "mean_angularity_beta_0p5",
            "mean_angularity_beta_1",
            "mean_angularity_beta_2",
            "mean_retained_top5",
            "mean_lost_top5",
        ]
    ].to_string(index=False)
)

print("\nTop-5 mean z values:")
for label, name in [(0, "gluon"), (1, "quark")]:
    sel = y == label
    print(
        f"{name:6s}: "
        + ", ".join(f"{v:.4f}" for v in z_top5[sel].mean(axis=0))
    )

print("\nFiles written:")
print(f"  Figures: {FIGURE_DIR}")
print(f"  Event data: {csv_path}")
print(f"  Summary: {ANALYSIS_DIR / 'qg_summary_by_class.csv'}")
print(f"  ML/QML arrays: {ANALYSIS_DIR / 'qg_top5_values.npz'}")

print("\nAnalysis finished successfully.")
