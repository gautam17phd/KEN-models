import numpy as np
import pandas as pd
import torch
from ase.neighborlist import neighbor_list
from sklearn.linear_model import Ridge
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


class HE26MolecularDataset(Dataset):

    def __init__(self, pickle_path):
        def canonical_atom_order(atoms):
            Z = atoms.get_atomic_numbers()
            pos = atoms.get_positions(wrap=True)          # resolves PBC image ambiguity
            centroid = pos.mean(axis=0, keepdims=True)
            d_centroid = np.linalg.norm(pos - centroid, axis=-1)

            # primary: species (keeps CNN channels chemically consistent across the dataset)
            # secondary: radial distance from centroid (rotation-invariant, no sign ambiguity)
            # np.lexsort ties resolve deterministically via original index as tertiary key
            order = np.lexsort((np.arange(len(Z)), d_centroid, Z))
            return order
        self.df = pd.read_pickle(pickle_path)

        print("🔍 Scanning HE26 chemical space for heavy elements & actinides...")
        all_atomic_numbers = set()
        for atoms in self.df["ase_atoms"]:
            all_atomic_numbers.update(atoms.get_atomic_numbers())

        self.unique_elements = sorted(list(all_atomic_numbers))
        self.num_elements = len(self.unique_elements)
        print(f"✅ Found {self.num_elements} unique elements.")

        # 1. Baseline Ridge Regression
        X = np.zeros((len(self.df), self.num_elements), dtype=np.float32)
        y = np.zeros(len(self.df), dtype=np.float32)
        z_to_idx = {z: idx for idx, z in enumerate(self.unique_elements)}

        for idx, row in self.df.iterrows():
            atoms = row["ase_atoms"]
            for z in atoms.get_atomic_numbers():
                X[idx, z_to_idx[z]] += 1.0
            y[idx] = float(row["energy"])

        print("🧮 Calculating isolated atomic baselines...")
        reg = Ridge(alpha=1.0, fit_intercept=False, solver="svd").fit(X, y)
        self.atomic_refs = {
            z: float(reg.coef_[z_to_idx[z]]) for z in self.unique_elements
        }

        # Pre-compute graph structure
        print("⚡ Pre-computing periodic graphs AND 2D CNN images...")
        self.precomputed_graphs = []

        for idx, row in self.df.iterrows():
            atoms = row["ase_atoms"]
            order = canonical_atom_order(atoms)
            atoms = atoms[order]
            num_atoms = atoms.get_global_number_of_atoms()

            # A. Periodic neighbor list
            i, j, D = neighbor_list("ijD", atoms, cutoff=5.0)

            # B. Vectorized Periodic Distance Matrix
            dist_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
            if len(i) > 0:
                dist_matrix[i, j] = np.linalg.norm(D, axis=-1)
            np.fill_diagonal(dist_matrix, 1.0)

            # C. Generate Electronic Spin Channels
            try:
                spin = np.array(atoms.get_initial_magnetic_moments())
            except Exception:
                spin = np.zeros(num_atoms)
            spin = np.nan_to_num(spin)

            spin_pair = np.tanh(spin[:, None] * spin[None, :])
            spin_dist = spin_pair * np.exp(-dist_matrix)
            diag_spin = np.zeros_like(spin_pair)
            np.fill_diagonal(diag_spin, spin)

            # Keep pointwise-only activation (no std-dev normalization).
            # Note: the std-dev-normalized version was ALSO exactly permutation-equivariant
            # (std over a multiset is order-invariant, per Lemma 1) — this change isn't
            # required for equivariance, it's just a simpler/batch-independent scaling.
            channels = np.stack(
                [np.tanh(spin_pair), np.tanh(spin_dist), np.tanh(diag_spin)],
                axis=0,
            )
            channels = torch.tensor(channels, dtype=torch.float32)  # restores the missing conversion

            # Store finished tensors
            self.precomputed_graphs.append({
                "edge_index": torch.tensor(np.stack([i, j]), dtype=torch.long),
                "edge_vec": torch.tensor(D, dtype=torch.float32),
                "cnn_channels": channels,
                "spin": torch.tensor(spin, dtype=torch.float32),
                # cache the reordered species/positions/forces so __getitem__
                # never falls back to the original ase_atoms order
                "atomic_numbers": torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
                "positions": torch.tensor(atoms.get_positions(), dtype=torch.float32),
                "forces": torch.tensor(row["forces"][order], dtype=torch.float32),
            })
        print("✅ Cache initialization complete.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache = self.precomputed_graphs[idx]

        energy_baseline = sum(self.atomic_refs[z] for z in cache["atomic_numbers"].tolist())
        energy_shifted = float(row["energy"]) - energy_baseline   # energy is order-independent, unaffected

        return {
            "energy": torch.tensor(energy_shifted, dtype=torch.float32),
            "forces": cache["forces"],
            "atomic_numbers": cache["atomic_numbers"],
            "positions": cache["positions"],
            "cache": cache,
        }


def hybrid_collate_fn(batch_list):
    pyg_data_list = []
    matrix_channels = []

    max_atoms = max(item["atomic_numbers"].size(0) for item in batch_list)

    for mol_idx, item in enumerate(batch_list):
        cache = item["cache"]
        num_atoms = item["atomic_numbers"].size(0)

        matrix_mapping = torch.stack(
            [
                torch.full((num_atoms,), mol_idx, dtype=torch.long),
                torch.arange(num_atoms, dtype=torch.long),
            ],
            dim=-1,
        )

        pos_tensor = item["positions"].clone().detach()

        g = Data(
            pos=pos_tensor,
            atomic_numbers=item["atomic_numbers"].clone().detach(),
            edge_index=cache["edge_index"].clone().detach(),
            edge_vec=cache["edge_vec"].clone().detach(),
            forces=item["forces"].clone().detach(),
            y=item["energy"].clone().detach().view(1),
            magmom=cache["spin"].clone().detach(),
            matrix_mapping=matrix_mapping,
        )
        pyg_data_list.append(g)

        # Zero-pad CNN grid canvas cleanly up to batch max_atoms
        padded = torch.zeros((3, max_atoms, max_atoms), dtype=torch.float32)
        padded[:, :num_atoms, :num_atoms] = cache["cnn_channels"]
        matrix_channels.append(padded)

    batched_graph = Batch.from_data_list(pyg_data_list)
    batched_graph.matrix_images = torch.stack(matrix_channels, dim=0)

    return batched_graph
