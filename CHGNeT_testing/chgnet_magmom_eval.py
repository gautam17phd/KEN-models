"""
Evaluate CHGNet magnetic moment (magmom) predictions on an extended XYZ dataset.

Usage:
    python chgnet_magmom_eval.py val.xyz --ref-key REF_magmoms --out results.csv

Requirements:
    pip install chgnet ase numpy pandas

Notes on CHGNet magmom output:
    - CHGNet's CHGNetCalculator / model.predict_structure returns a dict with
      keys like 'e', 'f', 's', 'm' when predict_magmom=True (site-projected
      magnetic moments, in mu_B, one value per atom).
    - If your reference magmoms live under a different extended-xyz per-atom
      array name (e.g. 'initial_magmoms', 'magmoms', 'REF_magmom'), pass it
      via --ref-key, or the script will try common names automatically.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_structures(xyz_path):
    from ase.io import read
    atoms_list = read(xyz_path, index=":")
    if isinstance(atoms_list, list):
        return atoms_list
    return [atoms_list]


def get_reference_magmoms(atoms, ref_key=None):
    """Try to pull per-atom reference magmoms out of an ASE Atoms object.

    ASE's extxyz reader treats recognized property names (energy, forces,
    stress, magmoms, charges, etc.) as calculator results attached via a
    SinglePointCalculator, NOT as generic atoms.arrays entries. So we must
    check atoms.get_magnetic_moments() (or atoms.calc.results['magmoms'])
    first, before falling back to arrays/info lookups.
    """
    # 1. Calculator-attached per-atom magmoms (most common for extxyz)
    if atoms.calc is not None:
        try:
            m = atoms.get_magnetic_moments()
            if m is not None and len(m) == len(atoms):
                return np.asarray(m, dtype=float)
        except Exception:
            pass
        # some files store a single scalar total magmom instead
        try:
            atoms.calc.results.get("magmom")
        except Exception:
            pass

    # 2. Explicit per-atom array under a custom or common name
    candidates = [ref_key] if ref_key else []
    candidates += ["REF_magmoms", "initial_magmoms", "magmoms", "magmom", "magnetic_moments"]
    for key in candidates:
        if key is None:
            continue
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=float)

    # 3. ASE's initial_magnetic_moments (set explicitly, not calculated)
    try:
        m = atoms.get_initial_magnetic_moments()
        if m is not None and np.any(m != 0):
            return np.asarray(m, dtype=float)
    except Exception:
        pass

    return None


def ase_atoms_to_pymatgen(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz_path", type=str, help="Path to extended XYZ file")
    parser.add_argument("--ref-key", type=str, default=None,
                         help="Per-atom array name in the XYZ holding reference magmoms")
    parser.add_argument("--out", type=str, default="chgnet_magmom_results.csv",
                         help="Where to write per-structure results CSV")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N structures (for a quick test)")
    parser.add_argument("--model", type=str, default=None,
                         help="Path/name of a CHGNet checkpoint; default = pretrained")
    args = parser.parse_args()

    try:
        from chgnet.model import CHGNet
    except ImportError:
        sys.exit("chgnet is not installed. Run: pip install chgnet")

    xyz_path = Path(args.xyz_path)
    if not xyz_path.exists():
        sys.exit(f"File not found: {xyz_path}")

    print(f"Loading structures from {xyz_path} ...")
    atoms_list = load_structures(xyz_path)
    if args.limit:
        atoms_list = atoms_list[: args.limit]
    print(f"Loaded {len(atoms_list)} structures")

    print("Loading CHGNet model ...")
    model = CHGNet.load(model_name=args.model) if args.model else CHGNet.load()

    # CHGNet's pretrained magmom head only has weights for Z=1..94 (H..Pu),
    # since Materials Project has almost no data past plutonium.
    CHGNET_MAX_Z = 94

    rows = []
    all_pred, all_ref = [], []
    n_no_ref = 0
    n_unsupported = 0
    unsupported_elements = set()

    for i, atoms in enumerate(atoms_list):
        max_z = int(atoms.get_atomic_numbers().max())
        if max_z > CHGNET_MAX_Z:
            from ase.data import chemical_symbols
            bad_symbols = sorted({
                chemical_symbols[z] for z in atoms.get_atomic_numbers() if z > CHGNET_MAX_Z
            })
            unsupported_elements.update(bad_symbols)
            n_unsupported += 1
            print(f"[{i}] skipped: contains {bad_symbols} (Z > {CHGNET_MAX_Z}), "
                  f"outside CHGNet's pretrained magmom support")
            continue

        try:
            structure = ase_atoms_to_pymatgen(atoms)
        except Exception as e:
            print(f"[{i}] failed to convert to pymatgen Structure: {e}")
            continue

        try:
            pred = model.predict_structure(structure, task="efsm")
        except Exception as e:
            print(f"[{i}] CHGNet prediction failed: {e}")
            continue

        pred_magmom = pred.get("m")
        if pred_magmom is None:
            print(f"[{i}] no magmom returned by model (task may need 'm' in requested outputs)")
            continue
        pred_magmom = np.asarray(pred_magmom, dtype=float).reshape(-1)

        ref_magmom = get_reference_magmoms(atoms, args.ref_key)

        row = {
            "index": i,
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "pred_magmom_mean": float(np.mean(pred_magmom)),
            "pred_magmom_absmean": float(np.mean(np.abs(pred_magmom))),
            "pred_magmom_max": float(np.max(np.abs(pred_magmom))),
        }

        if ref_magmom is not None and len(ref_magmom) == len(pred_magmom):
            diff = pred_magmom - ref_magmom
            row["mae"] = float(np.mean(np.abs(diff)))
            row["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
            row["ref_magmom_absmean"] = float(np.mean(np.abs(ref_magmom)))
            all_pred.append(pred_magmom)
            all_ref.append(ref_magmom)
        else:
            n_no_ref += 1
            row["mae"] = np.nan
            row["rmse"] = np.nan

        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(atoms_list)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote per-structure results to {args.out}")

    if all_pred:
        pred_flat = np.concatenate(all_pred)
        ref_flat = np.concatenate(all_ref)
        overall_mae = np.mean(np.abs(pred_flat - ref_flat))
        overall_rmse = np.sqrt(np.mean((pred_flat - ref_flat) ** 2))
        print("\n=== Overall (atom-level, structures with reference magmoms) ===")
        print(f"  structures with reference : {len(all_pred)} / {len(atoms_list)}")
        print(f"  total atoms compared      : {len(pred_flat)}")
        print(f"  MAE                       : {overall_mae:.4f} mu_B")
        print(f"  RMSE                      : {overall_rmse:.4f} mu_B")

        raw_out = str(Path(args.out).with_suffix("")) + "_raw.npz"
        np.savez(raw_out, pred=pred_flat, ref=ref_flat)
        print(f"  Saved raw per-atom pred/ref arrays to {raw_out} (for plotting)")
    else:
        print("\nNo reference magmoms found in the file (checked common array names).")
        print("Pass --ref-key <name> if your reference values are under a custom field.")

    if n_no_ref:
        print(f"\n{n_no_ref} structures had no usable reference magmoms and were skipped for error stats.")

    if n_unsupported:
        print(f"\n{n_unsupported} structures skipped entirely: contain elements beyond "
              f"CHGNet's pretrained support (Z > {CHGNET_MAX_Z}).")
        print(f"Unsupported elements found: {sorted(unsupported_elements)}")
        print("CHGNet's magmom head has no weights for these — this is a model coverage "
              "limit (Materials Project training data), not something fixable via this script.")


if __name__ == "__main__":
    main()
