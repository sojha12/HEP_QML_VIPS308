# HEP_QML_VIPS308
from pathlib import Path

readme = r"""# QML-HEPGraphs

> **Reproducing jet-tagging with classical ML, graph neural networks, and quantum graph neural networks.**

This project is a hands-on reproduction and extension of a recent jet-tagging study using **EnergyFlow / Pythia-generated jet data**, classical machine learning, graph representations, and eventually a **Qiskit-based Quantum Graph Neural Network (QGNN)**.

The immediate goal is **paper reproduction first**. Heavy-flavor tagging and other extensions will come later.

---

## Project Roadmap

```text
                    PAPER REPRODUCTION
                           │
                           ▼
                  ┌─────────────────┐
                  │   Jet Analysis  │
                  │ EnergyFlow/Data │
                  └────────┬────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Physics Observables │
                │ z, η, φ, ΔR, girth, │
                │ jet charge, etc.    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Dataset Preparation │
                │ truncation / labels │
                │ train-test split    │
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       CLASSICAL ML               GRAPH REPRESENTATION
       ┌─────────────┐             ┌───────────────┐
       │    MLP      │             │ Nodes / Edges │
       │    PFN      │             │ Symmetries    │
       └──────┬──────┘             └───────┬───────┘
              │                            │
              └────────────┬───────────────┘
                           ▼
                     GRAPH NN BASELINE
                           │
                           ▼
                    ┌───────────────┐
                    │    QGNN       │
                    │    Qiskit     │
                    └───────┬───────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Benchmark & Compare │
                 │ AUC / ROC / SIC     │
                 │ accuracy / scaling  │
                 └─────────────────────┘
```

---

# 1. Goals

The project is organized around three stages:

### Stage 1 — Reproduce the physics analysis

- Load the relevant jet datasets.
- Understand the event and particle representation.
- Reproduce the basic jet distributions.
- Calculate the observables used in the paper.
- Study constituent truncation.
- Reproduce the relevant baseline observables.

### Stage 2 — Build classical ML / graph baselines

- Prepare exactly the same particle-level inputs used by the paper.
- Build a simple MLP baseline.
- Build a permutation-aware Particle Flow Network (PFN).
- Convert jets into node/edge graph representations.
- Test a classical Graph Neural Network.
- Benchmark all models using the same data splits and metrics.

### Stage 3 — Build the QGNN

- Understand the graph structure used by the paper.
- Map particles/nodes to qubits.
- Implement data encoding.
- Implement shared node gates.
- Implement shared edge gates.
- Preserve permutation symmetry.
- Implement invariant measurement/aggregation.
- Reproduce the QGNN in Qiskit.
- Compare QGNN performance with the classical models.

---

# 2. Paper Reproduction

The first priority is to reproduce the paper before adding new datasets or physics tasks.

## Tasks

### Quark vs. Gluon

The Q/G task uses the EnergyFlow quark/gluon dataset.

For the model input, the project uses the five hardest constituents:

$$
N = 5
$$

with per-particle features

$$
(z,\eta,\phi).
$$

Therefore the model input has the structure

```text
Jet
├── particle 1 → (z, η, φ)
├── particle 2 → (z, η, φ)
├── particle 3 → (z, η, φ)
├── particle 4 → (z, η, φ)
└── particle 5 → (z, η, φ)
```

The resulting numerical representation is:

```text
X_QG.shape = (N_jets, 5, 3)
```

The jet-level label is stored separately:

```text
y_QG.shape = (N_jets,)
```

where `y[i]` is the class label for jet `i`.

---

### Up vs. Down Flavor

The flavor task uses ten hardest constituents:

$$
N = 10
$$

with particle features

$$
(z,\eta,\phi,\mathrm{PID},q).
$$

Representation:

```text
Jet
├── particle 1  → (z, η, φ, PID, q)
├── particle 2  → (z, η, φ, PID, q)
├── ...
└── particle 10 → (z, η, φ, PID, q)
```

The resulting input has the structure:

```text
X_UD.shape = (N_jets, 10, 5)
```

The flavor label is stored separately:

```text
y_UD.shape = (N_jets,)
```

The preprocessing used for the reproduction should remain faithful to the paper. In particular, PID encoding and angular preprocessing should be kept separate from the physics observables calculated later for interpretation.

---

# 3. Phase I — Jet Analysis

Before training anything, understand the data.

## Dataset inspection

- [x] Download/load EnergyFlow Q/G data
- [x] Verify number of jets
- [x] Verify particle dimensions
- [x] Inspect class labels
- [x] Inspect constituent multiplicity
- [x] Check missing/padded constituents
- [ ] Load/reproduce flavor dataset
- [ ] Verify flavor labels
- [ ] Verify particle feature encoding

## Initial plots

Generate:

- [x] Class balance
- [x] Constituent multiplicity
- [x] Constituent `z`
- [x] Constituent `eta`
- [x] Constituent `phi`
- [ ] Jet-level $p_T$
- [ ] Jet mass
- [ ] $\Delta R$ distributions
- [ ] Leading-constituent distributions

---

# 4. Phase II — Physics Observables

The goal here is to understand what information is actually present in the jets before giving it to a model.

## Q/G observables

### Jet Girth

Calculate

$$
g_J =
\sum_i
\frac{p_{T,i}}{p_{T,J}}
\Delta R_i
$$

with

$$
\Delta R_i =
\sqrt{
(\Delta\eta_i)^2+
(\Delta\phi_i)^2
}.
$$

Generate:

- [ ] Girth distribution for quarks
- [ ] Girth distribution for gluons
- [ ] Girth-based classifier
- [ ] ROC curve
- [ ] SIC curve

### Generalized angularities

Investigate the generalized angularity family and determine which observables are useful for interpreting the Q/G classifier.

- [ ] Implement angularity calculation
- [ ] Scan relevant exponents
- [ ] Plot quark/gluon distributions
- [ ] Compare observable performance

---

## u/d observables

### Jet Charge

Calculate

$$
Q_\kappa =
\sum_i q_i z_i^\kappa.
$$

Instead of assuming one value of $\kappa$, scan a range.

- [ ] Implement jet charge
- [ ] Scan $\kappa$
- [ ] Plot charge distributions
- [ ] Compare u/d separation
- [ ] Generate ROC
- [ ] Generate SIC
- [ ] Compare with learned model output

---

# 5. Phase III — Constituent Truncation

A major part of the reproduction is understanding what happens when only the hardest particles are retained.

For a jet with constituent momentum fractions $z_i$:

$$
f_{\mathrm{retained}}(N)
=
\sum_{i=1}^{N}z_i
$$

and

$$
f_{\mathrm{lost}}(N)
=
1-\sum_{i=1}^{N}z_i.
$$

Study:

```text
N = 1
N = 2
N = 3
...
N = 10
...
```

Then specifically investigate:

```text
Q/G → N = 5
u/d → N = 10
```

### Plots

- [ ] Mean retained $p_T$ vs. $N$
- [ ] Mean lost $p_T$ vs. $N$
- [ ] Distribution of retained momentum
- [ ] Distribution of lost momentum
- [ ] Comparison between classes
- [ ] Performance vs. number of constituents

This is important because constituent truncation is also what makes the eventual qubit mapping computationally manageable.

---

# 6. Phase IV — Prepare the ML Dataset

Create one standardized preprocessing pipeline so every model sees the same physics inputs.

```text
Raw dataset
     │
     ▼
Sort constituents by pT
     │
     ▼
Select N hardest particles
     │
     ▼
Select paper-defined features
     │
     ▼
Apply preprocessing
     │
     ▼
Train / validation / test split
     │
     ├──→ Classical ML
     ├──→ Graph NN
     └──→ QGNN
```

## Requirements

- [ ] Fixed random seed
- [ ] Reproducible train/test split
- [ ] Fit normalization only on training data
- [ ] Save processed arrays
- [ ] Save labels
- [ ] Save preprocessing metadata
- [ ] Save feature names
- [ ] Save dataset dimensions
- [ ] Verify no train/test leakage

Recommended outputs:

```text
data/processed/
├── qg_5particle.npz
├── ud_10particle.npz
├── preprocessing.json
└── dataset_summary.json
```

---

# 7. Phase V — Classical ML Baseline

Before building the quantum model, establish a strong classical reference.

## Model 1 — MLP

Start with a simple baseline.

```text
Particle features
       ↓
Flatten
       ↓
Dense
       ↓
ReLU
       ↓
Dense
       ↓
ReLU
       ↓
Output
```

For Q/G:

```text
5 particles × 3 features = 15 inputs
```

For u/d:

```text
10 particles × 5 features = 50 inputs
```

This model is intentionally simple.

Its purpose is to establish whether the representation contains enough information for ordinary supervised learning.

---

# 8. Phase VI — Particle Flow Network

The main classical benchmark should be permutation-aware.

The PFN has the basic structure

$$
f(x_1,\ldots,x_N)
=
F\left(
\sum_i \Phi(x_i)
\right).
$$

```text
particle 1 ── Φ ──┐
particle 2 ── Φ ──┤
particle 3 ── Φ ──┼── SUM ── F ── prediction
particle 4 ── Φ ──┤
particle 5 ── Φ ──┘
```

The sum makes the model invariant under permutations of the particles.

This is especially useful as a classical comparison to the permutation-symmetric QGNN.

### PFN tasks

- [ ] Implement $\Phi$
- [ ] Implement symmetric aggregation
- [ ] Implement $F$
- [ ] Train Q/G model
- [ ] Train u/d model
- [ ] Save model checkpoints
- [ ] Save training history
- [ ] Evaluate on held-out data

---

# 9. Phase VII — Graph Representation

Now convert each jet into a graph.

## Nodes

Each particle becomes a node:

```text
node_i = particle_i
```

with features such as:

```text
Q/G:
(z, η, φ)

u/d:
(z, η, φ, PID, q)
```

## Edges

For the graph representation, investigate relationships between particles:

$$
e_{ij}
=
f(x_i,x_j).
$$

Useful geometric quantities include:

$$
\Delta\eta_{ij},
\qquad
\Delta\phi_{ij},
\qquad
\Delta R_{ij}.
$$

For the paper reproduction, however, distinguish these graph/analysis quantities from the exact features actually uploaded to the QGNN.

### Graph tasks

- [ ] Create node arrays
- [ ] Create edge lists
- [ ] Create edge features
- [ ] Visualize representative jets
- [ ] Verify permutation behavior
- [ ] Save graph datasets
- [ ] Test graph batching
- [ ] Build classical GraphNN baseline

---

# 10. Phase VIII — Classical Graph Neural Network

Build a classical GraphNN before moving to quantum hardware/simulation.

```text
Jet
 │
 ▼
Nodes + Edges
 │
 ▼
Message Passing
 │
 ▼
Node Updates
 │
 ▼
Aggregation
 │
 ▼
Jet Representation
 │
 ▼
Classifier
```

Questions to investigate:

- Does explicitly using particle relationships improve performance?
- How does it compare with the PFN?
- Does the graph preserve permutation invariance?
- How sensitive is performance to the number of constituents?
- Which node/edge features matter?

---

# 11. Phase IX — QGNN

Only after the classical baselines are working.

The target structure is approximately:

```text
Particle features
       │
       ▼
   Data upload
       │
       ▼
 ┌───────────────────┐
 │ Shared node gates │
 │ Shared edge gates │
 └───────────────────┘
       │
       ▼
      repeat
       │
       ▼
 Measure <Zi>
       │
       ▼
 Mean / symmetric aggregation
       │
       ▼
   Classifier
```

### Quantum components

- [ ] Learn the paper's graph construction
- [ ] Understand node gates
- [ ] Understand edge gates
- [ ] Understand parameter sharing
- [ ] Understand data re-uploading
- [ ] Implement small circuit
- [ ] Verify circuit numerically
- [ ] Verify permutation symmetry
- [ ] Train on simulator
- [ ] Compare against classical baselines

### Qiskit

Initial target:

```text
Q/G:
N = 5 qubits

u/d:
N = 10 qubits
```

Start smaller for debugging if necessary:

```text
N = 3
```

Then scale up once the circuit is verified.

---

# 12. Benchmarking

Every model should be evaluated using the same test set.

## Core metrics

### Accuracy

$$
\mathrm{Accuracy}
=
\frac{TP+TN}{TP+TN+FP+FN}
$$

### True positive rate

$$
TPR =
\frac{TP}{TP+FN}
$$

### False positive rate

$$
FPR =
\frac{FP}{FP+TN}
$$

### ROC / AUC

Generate ROC curves for every classifier.

### Significance improvement characteristic

$$
SIC =
\frac{\epsilon_S}
{\sqrt{\epsilon_B}}.
$$

---

# 13. Benchmark Table

This will become the main results table for the project.

| Model | Q/G Accuracy | Q/G AUC | Q/G SIC | u/d Accuracy | u/d AUC | u/d SIC | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Girth / Jet Charge | — | — | — | — | — | — | 🟡 |
| MLP | — | — | — | — | — | — | ⬜ |
| PFN | — | — | — | — | — | — | ⬜ |
| GraphNN | — | — | — | — | — | — | ⬜ |
| QGNN | — | — | — | — | — | — | ⬜ |

> Numbers should only be added after the corresponding experiment has actually been run.

---

# 14. Progress Report

This section is intended to function as a **living research notebook**.

Add figures as the analysis develops.

---

## 14.1 Dataset Inspection

### Quark / Gluon Dataset

![Q/G class balance](data/figures/qg_class_balance.png)

**What this shows**

- Number of jets in each class.
- Whether the dataset is balanced.
- Any preprocessing or loading issues discovered.

**Observation**

> Add observations here after generating the plot.

---

### Constituent Multiplicity

![Constituent multiplicity](data/figures/constituent_multiplicity.png)

**Observation**

> Add observations here.

---

## 14.2 Particle-Level Distributions

### Momentum Fraction $z$

![z distribution](data/figures/z_distribution.png)

**Observation**

> Add observations about the constituent momentum distribution and differences between classes.

---

### Pseudorapidity $\eta$

![eta distribution](data/figures/eta_distribution.png)

**Observation**

> Add observations here.

---

### Azimuthal Angle $\phi$

![phi distribution](data/figures/phi_distribution.png)

**Observation**

> Add observations here.

---

# 15. Progress — Physics Observables

## Jet Girth

![Jet girth](data/figures/jet_girth.png)

### Interpretation

Jet girth measures how widely the transverse momentum of a jet is distributed relative to the jet axis.

**Questions to investigate**

- Do quark and gluon jets have visibly different distributions?
- Does girth provide useful Q/G discrimination?
- How does it compare with the learned model?

---

## Jet Charge

![Jet charge](data/figures/jet_charge.png)

### Interpretation

Jet charge is calculated as

$$
Q_\kappa =
\sum_i q_i z_i^\kappa.
$$

**Questions to investigate**

- How does the distribution change with $\kappa$?
- Which $\kappa$ values provide the strongest u/d separation?
- How does jet charge compare with the QGNN output?

---

# 16. Progress — Constituent Truncation

![Momentum retention](data/figures/truncation.png)

### Questions

- How much jet momentum is retained by the five hardest constituents?
- How much information is lost?
- Is the loss different for the two classes?
- Does model performance change as $N$ increases?

---

# 17. Progress — Classical ML

## MLP

![MLP training](results/mlp/training_curve.png)

### Results

| Metric | Q/G | u/d |
|---|---:|---:|
| Accuracy | — | — |
| AUC | — | — |
| Maximum SIC | — | — |

---

## PFN

![PFN ROC](results/pfn/roc_curve.png)

### Results

| Metric | Q/G | u/d |
|---|---:|---:|
| Accuracy | — | — |
| AUC | — | — |
| Maximum SIC | — | — |

---

# 18. Progress — Graph Representation

## Example Jet Graph

![Example graph](data/figures/example_jet_graph.png)

### Representation

```text
Node
 ├── particle features
 └── particle identity

Edge
 ├── pairwise geometry
 └── particle-particle relationship
```

### Questions

- What information does the graph add?
- Is the graph representation permutation invariant?
- What information is redundant?
- How does the graph compare with the PFN representation?

---

# 19. Progress — QGNN

## Circuit

![QGNN circuit](results/qgnn/circuit.png)

### Current status

- [ ] Data encoding
- [ ] Node gates
- [ ] Edge gates
- [ ] Parameter sharing
- [ ] Data re-uploading
- [ ] Measurement
- [ ] Invariant aggregation
- [ ] Training
- [ ] Benchmarking

---

## QGNN Results

| Metric | Classical PFN | Classical GraphNN | QGNN |
|---|---:|---:|---:|
| Accuracy | — | — | — |
| AUC | — | — | — |
| Max SIC | — | — | — |
| Parameters | — | — | — |
| Qubits | — | — | — |

---

# 20. Interpretation / Physics Questions

The final report should not only ask **which model performs best**.

It should investigate **why**.

### Questions to answer

1. What information distinguishes quark and gluon jets?
2. What information distinguishes up and down jets?
3. How much information is contained in the hardest constituents?
4. What information is lost through truncation?
5. Why does the flavor task require PID and charge?
6. Why is permutation invariance important?
7. What does the graph representation add?
8. How does the PFN encode permutation invariance?
9. How does the QGNN encode permutation invariance?
10. What does the quantum circuit actually learn?
11. Which observables correlate with the learned classifier?
12. How does QGNN performance compare with classical models?
13. How does performance scale with the number of qubits/particles?
14. What is the computational cost of each approach?

---

# 21. Reproducibility Checklist

- [ ] Environment documented
- [ ] Package versions recorded
- [ ] Random seeds fixed
- [ ] Dataset versions recorded
- [ ] Dataset preprocessing documented
- [ ] Train/test split documented
- [ ] Model hyperparameters saved
- [ ] Training histories saved
- [ ] Figures generated automatically
- [ ] Results saved as CSV/JSON
- [ ] Qiskit version recorded
- [ ] Circuit diagrams saved

---

# 22. Project Structure

```text
QML-HEPGraphs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── graphs/
│   └── figures/
│
├── setup/
│   ├── Load_Energy_Flow.py
│   ├── Load_Flavor.py
│   ├── Jet_analysis.py
│   ├── Build_Observables.py
│   ├── Prepare_ML_Data.py
│   ├── Train_MLP.py
│   ├── Train_PFN.py
│   ├── Build_Jet_Graphs.py
│   ├── Train_GraphNN.py
│   └── Train_QGNN.py
│
├── results/
│   ├── mlp/
│   ├── pfn/
│   ├── graphnn/
│   └── qgnn/
│
├── report/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

# 23. Running the Project

Activate the project environment:

```bash
cd /Users/br-7274/Downloads/QML-HEPGraphs
source .venv/bin/activate
```

Verify the interpreter:

```bash
which python
python --version
```

Run the EnergyFlow loader:

```bash
python setup/Load_Energy_Flow.py
```

Run the jet analysis:

```bash
python setup/Jet_analysis.py
```

Prepare ML data:

```bash
python setup/Prepare_ML_Data.py
```

Train the classical baseline:

```bash
python setup/Train_MLP.py
```

Train the PFN:

```bash
python setup/Train_PFN.py
```

Build graphs:

```bash
python setup/Build_Jet_Graphs.py
```

Train the classical GraphNN:

```bash
python setup/Train_GraphNN.py
```

Finally, after the classical pipeline is validated:

```bash
python setup/Train_QGNN.py
```

---

# 24. Current Progress

| Component | Status |
|---|---|
| Environment / `.venv` | 🟢 Working |
| EnergyFlow installation | 🟢 Working |
| EnergyFlow Q/G dataset | 🟢 Loaded |
| Basic jet plots | 🟢 Started |
| Physics observables | 🟡 In progress |
| Truncation study | 🟡 In progress |
| ML preprocessing | 🟡 In progress |
| MLP baseline | ⬜ |
| PFN baseline | ⬜ |
| Graph representation | 🟡 In progress |
| Classical GraphNN | ⬜ |
| Qiskit setup | ⬜ |
| QGNN | ⬜ |
| Full benchmark | ⬜ |
| Paper comparison | ⬜ |

---

# 25. References

### Main paper

Add the paper citation here:

```text
[Paper]
<Insert full citation / arXiv link>
```

### EnergyFlow

- EnergyFlow: https://energyflow.network/
- EnergyFlow documentation: https://energyflow.network/docs/

---

## Research Philosophy

The goal of this project is not simply to get a classifier with a high score.

The progression is:

$$
\boxed{
\text{Physics}
\rightarrow
\text{Representation}
\rightarrow
\text{Classical ML}
\rightarrow
\text{Graph ML}
\rightarrow
\text{QML}
}
$$

At every stage, the question is:

> **What information about the jet is the model actually using?**

That is the main thread connecting the physics analysis, graph representation, classical ML, and QGNN work.
"""

path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(f"Created: {path}")
print(f"Size: {path.stat().st_size:,} bytes")
from pathlib import Path

path = Path("/mnt/data/README.md")
text = path.read_text(encoding="utf-8")

insert_after = """# 4. Phase II — Physics observables

The purpose of this stage is to understand what physical information is contained in the jets before asking a machine-learning model to learn the classification.

"""

new_section = r"""# 4.5 From EnergyFlow data to physics-observable graphs

A central part of this project is making the transition from the **raw EnergyFlow event representation** to physics quantities that can be plotted, checked, and eventually supplied to the ML model.

The workflow is:

```text
EnergyFlow dataset
       │
       ▼
Load jets + labels
       │
       ▼
Inspect particle representation
       │
       ▼
Sort constituents by hardness
       │
       ▼
Select N hardest constituents
       │
       ├───────────────┐
       ▼               ▼
Particle-level     Jet-level
quantities         observables
       │               │
       └───────┬───────┘
               ▼
       Save analysis data
               │
               ▼
          Generate plots
               │
               ▼
        Prepare ML inputs
```

## Accessing the EnergyFlow dataset

The Q/G analysis uses the `energyflow` Python package.

The loading script is:

```text
setup/Load_Energy_Flow.py
```

The script downloads/loads the EnergyFlow quark/gluon dataset and saves the data locally so that subsequent analysis scripts do not need to repeatedly download it.

The dataset loaded during the current analysis has:

```text
X shape: (5000, 139, 4)
y shape: (5000,)
```

This means that the loaded sample contains:

- `5000` jets.
- Up to `139` particle slots per jet.
- `4` stored particle-level quantities.
- One jet-level class label in `y` for each jet.

The important conceptual distinction is:

```text
X
│
├── jet 0
│   ├── particle 0
│   ├── particle 1
│   ├── ...
│   └── particle 138
│
├── jet 1
│   ├── particle 0
│   ├── particle 1
│   ├── ...
│   └── particle 138
│
└── ...
```

while

```text
y[0] → label for jet 0
y[1] → label for jet 1
...
```

The particle-level feature convention needs to be checked against the dataset metadata before assigning a physical interpretation to every column. In the current Q/G analysis, the quantities used for the paper reproduction are the constituent momentum fraction and angular coordinates, while PID/charge information belongs to the separate flavor-tagging dataset.

---

## Sorting the constituents

The ML experiment does not simply use the first five entries in the raw array.

The constituents need to be ordered by their hardness so that we can identify the:

```text
1st hardest particle
2nd hardest particle
3rd hardest particle
4th hardest particle
5th hardest particle
```

For the Q/G reproduction:

```text
N = 5
```

The resulting ML representation is:

```text
Jet
│
├── hardest constituent  → (z, η, φ)
├── 2nd hardest          → (z, η, φ)
├── 3rd hardest          → (z, η, φ)
├── 4th hardest          → (z, η, φ)
└── 5th hardest          → (z, η, φ)
```

This produces an array of the form:

```text
X_QG.shape = (N_jets, 5, 3)
```

The sorting step is important because the paper's truncated representation is intended to retain the most important/highest-momentum constituents rather than arbitrary entries from the padded event representation.

---

## Generating particle-level graphs

Before constructing the eventual ML representation, we generate plots showing the distributions of the individual particle quantities.

The analysis script is:

```text
setup/Jet_analysis.py
```

Examples include:

```text
data/figures/
├── qg_class_balance.png
├── constituent_multiplicity.png
├── z_distribution.png
├── eta_distribution.png
└── phi_distribution.png
```

These plots answer basic questions such as:

- How many constituents does a typical jet contain?
- How is the momentum fraction distributed?
- What is the angular distribution of constituents?
- Are there visible differences between the two classes?
- How much information is contained in individual particle coordinates?

These are **analysis plots**, not yet ML predictions.

---

## Constructing angular quantities

The particle coordinates can also be converted into relative angular quantities.

For two constituents:

$$
\Delta R_{ij}
=
\sqrt{
(\Delta\eta_{ij})^2+
(\Delta\phi_{ij})^2
}.
$$

When calculating $\Delta\phi$, the periodicity of the azimuthal angle must be respected. In other words, two particles close to the $-\pi/\pi$ boundary should not incorrectly appear to be separated by almost $2\pi$.

Conceptually:

```text
particle i → (η_i, φ_i)
particle j → (η_j, φ_j)
                 │
                 ▼
          Δη and wrapped Δφ
                 │
                 ▼
                ΔR
```

These pairwise quantities are useful later when the particles are interpreted as nodes in a graph.

---

## Generating jet-level observables

The particle-level information can be aggregated into quantities describing the entire jet.

Examples include:

### Constituent multiplicity

Count the active constituents in each jet:

$$
N_{\mathrm{constituents}}.
$$

This gives a distribution that can be plotted separately for the two classes.

---

### Retained momentum fraction

After sorting the constituents, calculate

$$
f_{\mathrm{retained}}(N)
=
\sum_{i=1}^{N} z_i.
$$

This tells us how much of the jet's momentum is represented by the hardest $N$ constituents.

The complementary quantity is:

$$
f_{\mathrm{lost}}(N)
=
1-f_{\mathrm{retained}}(N).
$$

This directly connects the physics analysis to the truncation used for the ML experiment.

---

### Jet girth

Using the constituent momentum fractions and angular distance from the jet axis:

$$
g_J =
\sum_i
\frac{p_{T,i}}{p_{T,J}}
\Delta R_i.
$$

The resulting values are stored per jet and plotted as distributions.

The goal is to determine whether a simple physical observable already contains useful class information before training a neural network.

---

### Angular observables

The same constituent representation can be used to calculate generalized angular observables.

These are useful because they allow us to compare:

```text
Physics observable
        vs.
Learned ML representation
```

rather than treating the neural network as a completely opaque classifier.

---

## Analysis-data outputs

The observable-building stage should produce machine-readable data as well as images.

Suggested structure:

```text
data/
├── raw/
│
├── processed/
│   ├── qg_5particle.npz
│   ├── observables.npz
│   └── dataset_summary.json
│
└── figures/
    ├── qg_class_balance.png
    ├── constituent_multiplicity.png
    ├── z_distribution.png
    ├── eta_distribution.png
    ├── phi_distribution.png
    ├── jet_girth.png
    └── truncation.png
```

The important idea is that the graph/observable images are **not the only output**. The numerical values used to make the graphs should also be retained so that the same data can later be passed into the ML pipeline.

---

# 5. Phase II — Physics observables

The purpose of this stage is to understand what physical information is contained in the jets before asking a machine-learning model to learn the classification.

"""

# The README already has the Phase II heading immediately after the original
# Phase I section. Insert the EnergyFlow section just before that heading.
marker = "# 4. Phase II — Physics observables\n"
if marker in text and "## Accessing the EnergyFlow dataset" not in text:
    text = text.replace(marker, new_section + marker, 1)

# Expand the progress-report section with a dedicated data-to-plot subsection.
progress_marker = """# 8. Progress report

This section is intended to function as a **living research notebook**.

As new plots are generated, add them here with a short explanation of what they show and what was learned.

"""

progress_addition = r"""## 8.0 EnergyFlow → analysis data → plots

The reproducible analysis path is:

```text
Load_Energy_Flow.py
        ↓
EnergyFlow X, y
        ↓
Jet_analysis.py
        ↓
basic particle / jet distributions
        ↓
Build_Observables.py
        ↓
girth / angularities / truncation / other observables
        ↓
saved numerical data
        ↓
figures in data/figures/
        ↓
Prepare_ML_Data.py
        ↓
ML-ready particle representation
```

Each figure in this report should correspond to numerical data generated from the same analysis pipeline.

### Data-access record

```text
Dataset: EnergyFlow quark/gluon
Loaded jets: 5000
Maximum particle slots: 139
Stored particle features: 4
Jet labels: y
Q/G ML truncation: 5 hardest constituents
```

### Analysis figures

| Figure | What it checks | Generated from |
|---|---|---|
| Class balance | Number of jets in each class | `y` |
| Constituent multiplicity | Number of particles per jet | `X` |
| $z$ distribution | Constituent momentum fractions | `X` |
| $\eta$ distribution | Constituent pseudorapidity | `X` |
| $\phi$ distribution | Constituent azimuth | `X` |
| $\Delta R$ distribution | Pairwise angular separation | `η`, `φ` |
| Jet girth | Radial momentum spread | `z`, angular coordinates |
| Truncation | Information retained by hardest $N$ particles | sorted `z` |

This section should grow as each analysis plot is generated and interpreted.

"""

if progress_marker in text and "## 8.0 EnergyFlow → analysis data → plots" not in text:
    text = text.replace(progress_marker, progress_marker + progress_addition, 1)

# Add an explicit ML transition before the existing Phase V section.
ml_marker = "# 7. Phase V — Classical Machine Learning\n"
ml_addition = r"""## 6.5 Analysis-to-ML handoff

The ML dataset should come directly from the same processed information used during the physics analysis.

```text
Raw EnergyFlow jet
        ↓
sort constituents
        ↓
select hardest N
        ↓
select paper features
        ↓
validate shapes / labels
        ↓
normalize using training set only
        ↓
save X_train, X_val, X_test
        ↓
classical ML
```

For the Q/G task, the target representation is:

```text
N = 5
features = (z, η, φ)
```

giving:

```text
(N_jets, 5, 3)
```

The numerical arrays used for training should be saved separately from the figures so that the ML results can always be traced back to the original EnergyFlow data and preprocessing steps.

"""

if ml_marker in text and "## 6.5 Analysis-to-ML handoff" not in text:
    text = text.replace(ml_marker, ml_addition + ml_marker, 1)

path.write_text(text, encoding="utf-8")
print(f"Updated {path}")
print(f"{len(text.splitlines())} lines")

