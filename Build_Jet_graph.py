from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
GRAPH_DIR = PROJECT_ROOT / 'data' / 'graphs'
FIGURE_DIR = PROJECT_ROOT / 'data' / 'figures'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_NODES = 5
EXAMPLE_JET = 0

X_PATH = DATA_DIR / 'qg_X.npy'
Y_PATH = DATA_DIR / 'qg_y.npy'
if not X_PATH.exists() or not Y_PATH.exists():
    raise FileNotFoundError('Run python setup/Load_Energy_Flow.py first.')

X = np.load(X_PATH)
labels = np.load(Y_PATH).astype(int)
pt = X[:, :, 0].astype(float)
y = X[:, :, 1].astype(float)
phi = X[:, :, 2].astype(float)
pid = X[:, :, 3].astype(float)
mask = pt > 0

jet_pt = np.sum(np.where(mask, pt, 0.0), axis=1)
px = np.sum(np.where(mask, pt*np.cos(phi), 0.0), axis=1)
py = np.sum(np.where(mask, pt*np.sin(phi), 0.0), axis=1)
pz = np.sum(np.where(mask, pt*np.sinh(y), 0.0), axis=1)
energy = np.sum(np.where(mask, pt*np.cosh(y), 0.0), axis=1)
jet_y = 0.5*np.log(np.maximum(energy+pz,1e-12)/np.maximum(energy-pz,1e-12))
jet_phi = np.arctan2(py, px)

sort_idx = np.argsort(-pt, axis=1)
pt_s = np.take_along_axis(pt, sort_idx, axis=1)
y_s = np.take_along_axis(y, sort_idx, axis=1)
phi_s = np.take_along_axis(phi, sort_idx, axis=1)
pid_s = np.take_along_axis(pid, sort_idx, axis=1)
mask_s = np.take_along_axis(mask, sort_idx, axis=1)

pt_sel = pt_s[:, :N_NODES]
y_sel = y_s[:, :N_NODES]
phi_sel = phi_s[:, :N_NODES]
pid_sel = pid_s[:, :N_NODES]
mask_sel = mask_s[:, :N_NODES]
z_sel = np.divide(pt_sel, jet_pt[:, None], out=np.zeros_like(pt_sel), where=jet_pt[:, None] > 0)

def wrap(d):
    return (d + np.pi) % (2*np.pi) - np.pi

dy = y_sel - jet_y[:, None]
dphi = wrap(phi_sel - jet_phi[:, None])
dr = np.sqrt(dy**2 + dphi**2)
node_features = np.stack([z_sel, dy, dphi, dr, pid_sel], axis=-1)
node_features *= mask_sel[..., None]

adjacency = np.ones((N_NODES, N_NODES), dtype=np.int8)
np.fill_diagonal(adjacency, 0)

dy_ij = y_sel[:, :, None] - y_sel[:, None, :]
dphi_ij = wrap(phi_sel[:, :, None] - phi_sel[:, None, :])
dr_ij = np.sqrt(dy_ij**2 + dphi_ij**2)
edge_features = np.stack([dy_ij, dphi_ij, dr_ij], axis=-1)
valid = mask_sel[:, :, None] & mask_sel[:, None, :]
edge_features *= valid[..., None]
edge_features *= adjacency[None, :, :, None]

np.savez_compressed(
    GRAPH_DIR / 'qg_graphs.npz',
    node_features=node_features.astype(np.float32),
    edge_features=edge_features.astype(np.float32),
    adjacency=adjacency,
    labels=labels,
    selected_pt=pt_sel.astype(np.float32),
    selected_z=z_sel.astype(np.float32),
    selected_pid=pid_sel.astype(np.float32),
)

metadata = {
    'representation': 'EnergyFlow jet constituent graph',
    'nodes_per_graph': N_NODES,
    'graph_type': 'complete',
    'node_features': ['z', 'delta_y', 'delta_phi', 'delta_R', 'pid'],
    'edge_features': ['delta_y_ij', 'delta_phi_ij', 'delta_R_ij'],
    'labels': {'0': 'gluon', '1': 'quark'},
}
with open(GRAPH_DIR / 'graph_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

if not 0 <= EXAMPLE_JET < len(X):
    raise ValueError('EXAMPLE_JET is outside the dataset.')

node_csv = GRAPH_DIR / 'example_jet_nodes.csv'
with open(node_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['node','label','pt_GeV','z','delta_y','delta_phi','delta_R','pid'])
    for i in range(N_NODES):
        w.writerow([i, 'quark' if labels[EXAMPLE_JET] == 1 else 'gluon', pt_sel[EXAMPLE_JET,i], z_sel[EXAMPLE_JET,i], dy[EXAMPLE_JET,i], dphi[EXAMPLE_JET,i], dr[EXAMPLE_JET,i], pid_sel[EXAMPLE_JET,i]])

edge_csv = GRAPH_DIR / 'example_jet_edges.csv'
with open(edge_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['source','target','delta_y','delta_phi','delta_R'])
    for i in range(N_NODES):
        for j in range(N_NODES):
            if adjacency[i,j]:
                w.writerow([i,j,edge_features[EXAMPLE_JET,i,j,0],edge_features[EXAMPLE_JET,i,j,1],edge_features[EXAMPLE_JET,i,j,2]])

# Example graph visualization in jet-relative (y, phi) coordinates.
plt.figure(figsize=(8,7))
for i in range(N_NODES):
    for j in range(i+1, N_NODES):
        plt.plot([dy[EXAMPLE_JET,i],dy[EXAMPLE_JET,j]], [dphi[EXAMPLE_JET,i],dphi[EXAMPLE_JET,j]], linewidth=0.8, alpha=0.25)
plt.scatter(dy[EXAMPLE_JET], dphi[EXAMPLE_JET], s=1000*z_sel[EXAMPLE_JET]+25, alpha=0.85)
for i in range(N_NODES):
    plt.annotate(str(i+1), (dy[EXAMPLE_JET,i], dphi[EXAMPLE_JET,i]), xytext=(5,5), textcoords='offset points')
plt.xlabel(r'$\Delta y$ relative to jet axis')
plt.ylabel(r'$\Delta\phi$ relative to jet axis')
plt.title(f"Example jet graph — {'Quark' if labels[EXAMPLE_JET] == 1 else 'Gluon'}")
plt.axhline(0, linewidth=0.6, alpha=0.3)
plt.axvline(0, linewidth=0.6, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / '14_example_jet_graph.png', dpi=300, bbox_inches='tight')
plt.close()

print('='*70)
print('EnergyFlow -> Jet Graph Representation')
print('='*70)
print('Input X shape:', X.shape)
print('Graphs:', len(X))
print('Nodes per graph:', N_NODES)
print('Node feature shape:', node_features.shape)
print('Edge feature shape:', edge_features.shape)
print('Adjacency shape:', adjacency.shape)
print('\nNode features: z, delta_y, delta_phi, delta_R, pid')
print('Edge features: delta_y_ij, delta_phi_ij, delta_R_ij')
print(f"Example jet class: {'quark' if labels[EXAMPLE_JET] == 1 else 'gluon'}")
print('Example top-N z:', np.array2string(z_sel[EXAMPLE_JET], precision=5))
print('Example retained fraction:', f"{z_sel[EXAMPLE_JET].sum():.5f}")
print('\nSaved:', GRAPH_DIR / 'qg_graphs.npz')
print('Saved:', GRAPH_DIR / 'graph_metadata.json')
print('Saved:', node_csv)
print('Saved:', edge_csv)
print('Saved:', FIGURE_DIR / '14_example_jet_graph.png')
