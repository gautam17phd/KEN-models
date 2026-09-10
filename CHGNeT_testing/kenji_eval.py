"""
Evaluate the custom CompiledMLIPCalculator (KENJI) on a pandas-pickle test set
produced by the extxyz -> pkl conversion pipeline (columns: ase_atoms, energy,
forces, structure_name, optionally stress).

IMPORTANT — read before trusting results:
    The 'ase_atoms' objects in this pkl have their initial_magnetic_moments set
    to the SAME per-atom magmom values used as ground truth (parsed straight off
    the xyz magmom column). Since CompiledMLIPCalculator.calculate() reads
    atoms.get_initial_magnetic_moments() as an INPUT feature (spin_t), running
    prediction directly on these atoms feeds the reference answer straight into
    the model. That is only a fair test if training did the same thing
    (self-consistent / teacher-forcing scheme). If training instead started from
    a zeroed or perturbed guess, you must test the same way, or your magmom
    "prediction" numbers are meaningless (near-perfect by construction).

    Use --magmom-mode to control this:
      real   : feed the true initial magmoms through as-is (matches file content)
      zero   : zero out all initial magmoms before prediction (blind test)
      random : replace with small random noise around zero (blind test, avoids
               degenerate all-zero input if your model behaves oddly at exactly 0)

    Run BOTH real and zero and compare — if they give wildly different magmom
    error, that tells you which regime your model was actually trained in.

Usage:
    python kenji_eval.py testing.pkl --model KENJI.ptc --device cuda --magmom-mode zero
"""

import argparse
import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.calculators.calculator import Calculator, all_properties
from ase.neighborlist import neighbor_list

# ============================================================
# PHYSICAL BASELINE VALUES INITIALIZATION
# ============================================================
ATOMIC_REFS = {
    8: -14.4981, 39: 3.8160, 40: 1.0342, 41: 2.1881,
    57: 5.6295,  58: 2.8697, 60: 5.8286, 63: 1.8170,
    64: -3.1563, 71: 5.7609, 90: 0.0736, 91: -2.4681,
    92: -3.1598, 93: -3.6084, 94: -4.5609, 95: -5.4599,
    96: -7.2799, 98: -6.5734
}
MAX_Z = max(ATOMIC_REFS.keys())
REF_LOOKUP = np.zeros(MAX_Z + 1, dtype=np.float32)
for z, ref_val in ATOMIC_REFS.items():
    REF_LOOKUP[z] = ref_val


class CompiledMLIPCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'magmoms']

    def __init__(self, compiled_model_path, device=torch.device("cpu"), r_cut=7.0):
        super().__init__()
        print(f"Loading pre-compiled graph into memory stream: {compiled_model_path}")
        cpu_model = torch.jit.load(compiled_model_path, map_location="cpu")
        self.model = cpu_model.float().eval()
        self.device = device
        self.r_cut = r_cut

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_properties):
        super().calculate(atoms, properties, system_changes)

        # ============================================================
        # STEP 0: Canonical atom reordering — MUST match
        # HE26MolecularDataset.canonical_atom_order() exactly, since the
        # CNN's matrix_images grid relies on this ordering for spatial
        # locality (species-sorted, then radial distance from centroid).
        # ============================================================
        Z_orig = self.atoms.get_atomic_numbers()
        pos_orig_wrapped = self.atoms.get_positions(wrap=True)
        centroid = pos_orig_wrapped.mean(axis=0, keepdims=True)
        d_centroid = np.linalg.norm(pos_orig_wrapped - centroid, axis=-1)
        order = np.lexsort((np.arange(len(Z_orig)), d_centroid, Z_orig))
        inverse_order = np.argsort(order)  # to map outputs back to original atom order

        atoms_reordered = self.atoms[order]
        num_atoms = len(atoms_reordered)

        positions = torch.tensor(atoms_reordered.get_positions(), dtype=torch.float32, device=self.device)
        positions.requires_grad_(True)

        atomic_numbers = atoms_reordered.get_atomic_numbers()

        # ============================================================
        # STEP 1: Graph edges — same cutoff, same 'ijD' convention as dataset
        # ============================================================
        idx_i, idx_j, displacements = neighbor_list('ijD', atoms_reordered, self.r_cut)
        edge_index = torch.tensor(np.stack([idx_i, idx_j]), dtype=torch.long, device=self.device)
        edge_vec = torch.tensor(displacements, dtype=torch.float32, device=self.device)

        dr_open = positions[edge_index[1]] - positions[edge_index[0]]
        with torch.no_grad():
            cell_offsets = edge_vec - dr_open
        dynamic_edge_vec = dr_open + cell_offsets

        # ============================================================
        # STEP 2: dist_matrix — RAW distances, diagonal hard-set to 1.0.
        # No cosine envelope here: the dataset never applied one.
        # ============================================================
        distances = torch.norm(dynamic_edge_vec, dim=-1)
        dist_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float32, device=self.device)
        if len(idx_i) > 0:
            dist_matrix[edge_index[0], edge_index[1]] = distances
        dist_matrix.fill_diagonal_(1.0)

        # ============================================================
        # STEP 3: Spin channels — matches dataset exactly: single tanh per
        # channel, NO std-normalization, NO extra outer tanh.
        # ============================================================
        try:
            spin_np = np.array(atoms_reordered.get_initial_magnetic_moments())
        except Exception:
            spin_np = np.zeros(num_atoms)
        spin_np = np.nan_to_num(spin_np)

        spin_t = torch.tensor(spin_np, dtype=torch.float32, device=self.device)

        spin_pair = torch.tanh(spin_t[:, None] * spin_t[None, :])
        spin_dist = spin_pair * torch.exp(-dist_matrix)
        diag_spin = torch.zeros_like(spin_pair)
        diag_spin.diagonal().copy_(spin_t)

        channels = torch.stack(
            [torch.tanh(spin_pair), torch.tanh(spin_dist), torch.tanh(diag_spin)],
            dim=0,
        )
        matrix_images = channels.unsqueeze(0)  # add batch dim (batch size 1 here)

        matrix_mapping = torch.stack([
            torch.zeros(num_atoms, device=self.device),
            torch.arange(num_atoms, device=self.device)
        ], dim=-1).long()

        batch_indices = torch.zeros(num_atoms, dtype=torch.long, device=self.device)
        atomic_numbers_tensor = torch.tensor(atomic_numbers, dtype=torch.long, device=self.device)

        pred_energy, pred_magmoms = self.model(
            positions,
            atomic_numbers_tensor,
            edge_index,
            dynamic_edge_vec,
            matrix_images,
            matrix_mapping,
            batch_indices
        )
        pred_energy_flat = pred_energy.view(-1)

        baseline_energy = REF_LOOKUP[atomic_numbers].sum()
        total_energy_ev = pred_energy_flat.item() + baseline_energy

        forces_grad_tuple = torch.autograd.grad(
            outputs=pred_energy_flat,
            inputs=positions,
            grad_outputs=torch.ones_like(pred_energy_flat),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )

        self.results['energy'] = total_energy_ev

        forces_canonical = -forces_grad_tuple[0].detach().cpu().numpy().astype(np.float64)
        pred_magmoms_canonical = pred_magmoms.detach().cpu().numpy().astype(np.float64).reshape(-1)
        if pred_magmoms_canonical.shape[0] != num_atoms:
            warnings.warn(
                f"Predicted magmom array shape {pred_magmoms_canonical.shape} doesn't match "
                f"num_atoms={num_atoms}; check model output ordering."
            )

        # Map outputs back from canonical (species, radial-distance) order to
        # the ORIGINAL atom order the caller passed in — everything above ran
        # in canonical order to match training, but callers (and any
        # reference arrays you compare against) expect original ordering.
        self.results['forces'] = forces_canonical[inverse_order]
        self.results['magmoms'] = pred_magmoms_canonical[inverse_order]


def load_calculator(model_path, device_str, r_cut):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return CompiledMLIPCalculator(model_path, device=device, r_cut=r_cut)


def apply_magmom_mode(atoms, mode, rng, perturb_scale):
    ref_magmoms = np.array(atoms.get_initial_magnetic_moments(), dtype=float).copy()
    if mode == "real":
        pass  # leave as-is
    elif mode == "zero":
        atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))
    elif mode == "random":
        # small perturbation around the TRUE value (not independent of it) —
        # tests whether the model can correct a noisy-but-informative guess
        noise = rng.normal(scale=perturb_scale, size=len(atoms))
        atoms.set_initial_magnetic_moments(ref_magmoms + noise)
    elif mode == "perturbed":
        # true value + noise: tests whether the model corrects a noisy guess
        # back toward the reference, which is a different (weaker) claim than
        # predicting from zero information.
        noise = rng.normal(scale=perturb_scale, size=len(atoms))
        atoms.set_initial_magnetic_moments(ref_magmoms + noise)
    else:
        raise ValueError(f"unknown magmom-mode: {mode}")
    return ref_magmoms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl_path", type=str)
    parser.add_argument("--model", type=str, default="KENJI.ptc")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rcut", type=float, default=5.0)
    parser.add_argument("--magmom-mode", choices=["real", "zero", "random", "perturbed"], default="real")
    parser.add_argument("--perturb-scale", type=float, default=0.3,
                         help="Std dev (mu_B) of Gaussian noise for 'random'/'perturbed' modes")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default="kenji_eval_results.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pkl_path = Path(args.pkl_path)
    if not pkl_path.exists():
        sys.exit(f"File not found: {pkl_path}")

    print(f"Loading {pkl_path} ...")
    df = pd.read_pickle(pkl_path)
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"Loaded {len(df)} structures")
    print(f"magmom-mode = {args.magmom_mode!r}  (see script docstring for what this means)")

    calc = load_calculator(args.model, args.device, args.rcut)
    rng = np.random.default_rng(args.seed)

    rows = []
    e_pred_all, e_ref_all = [], []
    f_pred_all, f_ref_all = [], []
    m_pred_all, m_ref_all = [], []

    for i, row in df.iterrows():
        atoms = copy.deepcopy(row["ase_atoms"])
        ref_energy = float(row["energy"])
        ref_forces = np.asarray(row["forces"], dtype=float)

        ref_magmoms = apply_magmom_mode(atoms, args.magmom_mode, rng, args.perturb_scale)

        atoms.calc = calc
        try:
            pred_energy = atoms.get_potential_energy()
            pred_forces = atoms.get_forces()
            pred_magmoms = np.asarray(atoms.calc.results.get("magmoms"), dtype=float)
        except Exception as e:
            print(f"[{i}] prediction failed: {e}")
            continue

        n_atoms = len(atoms)
        e_mae = abs(pred_energy - ref_energy)
        e_mae_per_atom = e_mae / n_atoms if n_atoms else np.nan

        f_diff = pred_forces - ref_forces
        f_mae = float(np.mean(np.abs(f_diff)))
        f_rmse = float(np.sqrt(np.mean(f_diff ** 2)))

        row_result = {
            "index": i,
            "structure_name": row.get("structure_name", None),
            "n_atoms": n_atoms,
            "energy_ref": ref_energy,
            "energy_pred": pred_energy,
            "energy_abs_err": e_mae,
            "energy_abs_err_per_atom": e_mae_per_atom,
            "force_mae": f_mae,
            "force_rmse": f_rmse,
        }

        if pred_magmoms is not None and pred_magmoms.shape[0] == n_atoms:
            m_diff = pred_magmoms - ref_magmoms
            row_result["magmom_mae"] = float(np.mean(np.abs(m_diff)))
            row_result["magmom_rmse"] = float(np.sqrt(np.mean(m_diff ** 2)))
            m_pred_all.append(pred_magmoms)
            m_ref_all.append(ref_magmoms)
        else:
            row_result["magmom_mae"] = np.nan
            row_result["magmom_rmse"] = np.nan

        rows.append(row_result)
        e_pred_all.append(pred_energy)
        e_ref_all.append(ref_energy)
        f_pred_all.append(pred_forces.flatten())
        f_ref_all.append(ref_forces.flatten())

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(df)}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.out, index=False)
    print(f"\nWrote per-structure results to {args.out}")

    if e_pred_all:
        e_pred_all = np.array(e_pred_all)
        e_ref_all = np.array(e_ref_all)
        print("\n=== Energy ===")
        print(f"  MAE  : {np.mean(np.abs(e_pred_all - e_ref_all)):.4f} eV")
        print(f"  RMSE : {np.sqrt(np.mean((e_pred_all - e_ref_all) ** 2)):.4f} eV")

    if f_pred_all:
        f_pred_flat = np.concatenate(f_pred_all)
        f_ref_flat = np.concatenate(f_ref_all)
        print("\n=== Forces (all components) ===")
        print(f"  MAE  : {np.mean(np.abs(f_pred_flat - f_ref_flat)):.4f} eV/A")
        print(f"  RMSE : {np.sqrt(np.mean((f_pred_flat - f_ref_flat) ** 2)):.4f} eV/A")

    if m_pred_all:
        m_pred_flat = np.concatenate(m_pred_all)
        m_ref_flat = np.concatenate(m_ref_all)
        print(f"\n=== Magmoms (mode={args.magmom_mode}) ===")
        print(f"  MAE  : {np.mean(np.abs(m_pred_flat - m_ref_flat)):.4f} mu_B")
        print(f"  RMSE : {np.sqrt(np.mean((m_pred_flat - m_ref_flat) ** 2)):.4f} mu_B")

        zero_baseline_rmse = np.sqrt(np.mean(m_ref_flat ** 2))
        zero_baseline_mae = np.mean(np.abs(m_ref_flat))
        print(f"\n  --- Trivial 'always predict zero' baseline (same reference set) ---")
        print(f"  baseline MAE  : {zero_baseline_mae:.4f} mu_B")
        print(f"  baseline RMSE : {zero_baseline_rmse:.4f} mu_B")
        model_rmse = np.sqrt(np.mean((m_pred_flat - m_ref_flat) ** 2))
        if model_rmse > 0.9 * zero_baseline_rmse:
            print("  WARNING: model RMSE is close to (or worse than) the zero-baseline.")
            print("           This suggests the model may be collapsing toward a near-")
            print("           constant output rather than genuinely predicting magmoms.")
        else:
            improvement = 100 * (1 - model_rmse / zero_baseline_rmse)
            print(f"  Model beats zero-baseline by {improvement:.1f}% (RMSE reduction).")

        raw_out = str(Path(args.out).with_suffix("")) + f"_{args.magmom_mode}_raw.npz"
        np.savez(raw_out, pred=m_pred_flat, ref=m_ref_flat)
        print(f"  Saved raw per-atom pred/ref arrays to {raw_out} (for plotting)")
        if args.magmom_mode == "real":
            print("  NOTE: mode='real' feeds ground-truth magmoms as model input.")
            print("        Near-zero error here does NOT prove predictive skill —")
            print("        rerun with --magmom-mode zero to test blind prediction.")


if __name__ == "__main__":
    main()
