import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms, units
from ase.constraints import FixAtoms
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_properties
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.eos import EquationOfState

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

# ============================================================
# CPU-SAFE TORCHSCRIPT CALCULATOR WRAPPER
# ============================================================
# ============================================================
# CPU-SAFE TORCHSCRIPT CALCULATOR WRAPPER
# ============================================================
class CompiledMLIPCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, compiled_model_path, device=torch.device("cpu"), r_cut=5.0):
        super().__init__()
        # Load directly onto CPU to bypass serialization device barriers
        print(f"📦 Loading pre-compiled graph into memory stream: {compiled_model_path}")
        cpu_model = torch.jit.load(compiled_model_path, map_location="cpu")
        self.model = cpu_model.float().eval()
        self.device = device
        self.r_cut = r_cut 

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_properties):
        super().calculate(atoms, properties, system_changes)
        
        positions = torch.tensor(self.atoms.get_positions(), dtype=torch.float32, device=self.device)
        positions.requires_grad_(True)
        
        atomic_numbers = self.atoms.get_atomic_numbers()
        num_atoms = len(atomic_numbers)
        
        idx_i, idx_j, displacements = neighbor_list('ijD', self.atoms, self.r_cut)
        edge_index = torch.tensor(np.stack([idx_i, idx_j]), dtype=torch.long, device=self.device)
        edge_vec = torch.tensor(displacements, dtype=torch.float32, device=self.device)

        # Re-anchor gradient tracking graph safely
        dr_open = positions[edge_index[1]] - positions[edge_index[0]]
        with torch.no_grad():
            cell_offsets = edge_vec - dr_open
        dynamic_edge_vec = dr_open + cell_offsets

        # Vectorized CNN matrix image formulation
        dist_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        if len(idx_i) > 0:
            dist_matrix[idx_i, idx_j] = np.linalg.norm(displacements, axis=-1)
        np.fill_diagonal(dist_matrix, 1.0)

        try:
            spin = np.array(self.atoms.get_initial_magnetic_moments())
        except Exception:
            spin = np.zeros(num_atoms)
        spin = np.nan_to_num(spin)
        
        spin_pair = np.tanh(spin[:, None] * spin[None, :])
        spin_dist = spin_pair * np.exp(-dist_matrix)
        diag_spin = np.zeros_like(spin_pair)
        np.fill_diagonal(diag_spin, spin)

        spin_pair = np.tanh(spin_pair / (np.std(spin_pair) + 1e-8))
        spin_dist = np.tanh(spin_dist / (np.std(spin_dist) + 1e-8))
        diag_spin = np.tanh(diag_spin / (np.std(diag_spin) + 1e-8))
        
        channels = np.stack([spin_pair, spin_dist, diag_spin], axis=0)
        matrix_images = torch.tensor(np.tanh(channels), dtype=torch.float32, device=self.device).unsqueeze(0)

        matrix_mapping = torch.stack([
            torch.zeros(num_atoms, device=self.device), 
            torch.arange(num_atoms, device=self.device)
        ], dim=-1).long()
        
        batch_indices = torch.zeros(num_atoms, dtype=torch.long, device=self.device)
        atomic_numbers_tensor = torch.tensor(atomic_numbers, dtype=torch.long, device=self.device)

        pred_energy, _ = self.model(
            positions,               # Argument 1: pos
            atomic_numbers_tensor,   # Argument 2: atomic_numbers
            edge_index,              # Argument 3: edge_index
            dynamic_edge_vec,        # Argument 4: Added edge_vec here in slot 4!
            matrix_images,           # Argument 5: matrix_images
            matrix_mapping,          # Argument 6: matrix_mapping
            batch_indices            # Argument 7: batch_indices
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
        self.results['forces'] = -forces_grad_tuple[0].detach().cpu().numpy().astype(np.float64)

# ============================================================
# MASTER UNIFIED SIMULATION RUNNER PIPELINE
# ============================================================
def main():
    device = torch.device("cpu")
    print(f"🖥️ Target Compute Context Assigned: {device}")
    
    # Configure path pointing to your model file footprint
    calc = CompiledMLIPCalculator("KENJI.ptc", device, r_cut=5.0)
    
    # CRUCIAL: Point this directly to your downloaded database file name
    cif_file_path = "AmO2.cif"
    if not os.path.exists(cif_file_path):
        raise FileNotFoundError(f"❌ Missing mandatory CIF input path: {cif_file_path}. Please place it in this directory.")

    print(f"📖 Parsing experimental crystal lattice coordinates from database file: {cif_file_path}")
    unit_cell = read(cif_file_path)
    
    # Scale unit cell to establish standard 2x2x2 supercell dimension blocks
    bulk_amo2 = unit_cell * (2, 2, 2)
    bulk_amo2.set_pbc((True, True, True))
    bulk_amo2.set_initial_magnetic_moments(np.ones(len(bulk_amo2)) * 2.0)
    bulk_amo2.calc = calc

    # --------------------------------------------------------
    # STEP 1: INITIAL BFGS STRUCTURAL MINIMIZATION
    # --------------------------------------------------------
    print("\n⏳ Step 1: Running Initial Geometry Relaxation...")
    opt = BFGS(bulk_amo2, trajectory='relaxation_track.traj')
    opt.run(fmax=0.05, steps=500) 
    relaxed_cell = read('relaxation_track.traj', index=-1)
    print(f"📐 Relaxed Cell Volume: {relaxed_cell.get_volume():.3f} Å³")

    # --------------------------------------------------------
    # STEP 2: HIGH-TEMPERATURE LANGEVIN MOLECULAR DYNAMICS
    # --------------------------------------------------------
    print("\n🚀 Step 2: Launching Production Langevin MD (1400 K)...")
    md_atoms = relaxed_cell.copy()
    md_atoms.calc = calc
    
    def log_langevin_progress():
        epot = float(md_atoms.get_potential_energy())
        ekin = float(md_atoms.get_kinetic_energy())
        temp = ekin / (1.5 * units.kB * len(md_atoms))
        print(f"🕒 MD Step | Potential Energy: {epot:.4f} eV | Temp: {temp:.1f} K")

    dyn_langevin = Langevin(md_atoms, timestep=1.0 * units.fs, temperature_K=1400, friction=0.02, fixcm=False)
    traj_langevin = Trajectory('bulk_amo2_fluorite.traj', 'w', md_atoms)
    
    dyn_langevin.attach(traj_langevin, interval=2)
    dyn_langevin.attach(log_langevin_progress, interval=10)
    dyn_langevin.run(steps=4000) 
    print("🏁 Langevin MD complete. Saved to 'bulk_amo2_fluorite.traj'.")

    # --------------------------------------------------------
    # STEP 3: STATIC EQUATION OF STATE (EOS) DEFORMATION
    # --------------------------------------------------------
    print("\n📈 Step 3: Executing Equation of State (EOS) Scans...")
    volumes, energies = [], []
    scale_factors = np.linspace(0.90, 1.10, 15)
    
    for scale in scale_factors:
        test_atoms = relaxed_cell.copy()
        test_atoms.calc = calc
        test_atoms.set_cell(relaxed_cell.cell * scale, scale_atoms=True)
        
        volumes.append(test_atoms.get_volume())
        energies.append(test_atoms.get_potential_energy())

    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0, e0, B = eos.fit()
    print(f"✅ EOS Fitted Equilibrium Volume: {v0:.3f} Å³ | Bulk Modulus: {B / units.GPa:.2f} GPa")
    eos.plot('amo2_eos_curve.png')
    print("💾 Plot saved as 'amo2_eos_curve.png'")

    # --------------------------------------------------------
    # STEP 4: MICROCANONICAL ENSEMBLE (NVE) CONSERVATION TEST
    # --------------------------------------------------------
    print("\n⏱️ Step 4: Initiating NVE Stability Profile Check...")
    nve_atoms = relaxed_cell.copy()
    nve_atoms.calc = calc
    MaxwellBoltzmannDistribution(nve_atoms, temperature_K=600.0)
    
    dyn_nve = VelocityVerlet(nve_atoms, timestep=1.0 * units.fs)
    time_track, pot_track, kin_track, tot_track = [], [], [], []

    def gather_nve_metrics():
        epot = nve_atoms.get_potential_energy() / len(nve_atoms)
        ekin = nve_atoms.get_kinetic_energy() / len(nve_atoms)
        etot = epot + ekin
        
        pot_track.append(epot)
        kin_track.append(ekin)
        tot_track.append(etot)
        time_track.append(len(time_track))

    dyn_nve.attach(gather_nve_metrics, interval=1)
    dyn_nve.run(steps=200)

    plt.figure(figsize=(7, 4))
    plt.plot(time_track, pot_track, label='Potential Energy', color='blue', linestyle='--')
    plt.plot(time_track, kin_track, label='Kinetic Energy', color='orange', linestyle='--')
    plt.plot(time_track, tot_track, label='Total Energy', color='black', linewidth=2)
    plt.xlabel('Simulation Step (fs)')
    plt.ylabel('Energy (eV/atom)')
    plt.figure(figsize=(7, 4))
    plt.plot(time_track, pot_track, label='Potential Energy', color='blue', linestyle='--')
    plt.plot(time_track, kin_track, label='Kinetic Energy', color='orange', linestyle='--')
    plt.plot(time_track, tot_track, label='Total Energy', color='black', linewidth=2)
    plt.xlabel('Simulation Step (fs)')
    plt.ylabel('Energy (eV/atom)')
    plt.title('KEN Microcanonical Ensemble (NVE) Stability Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('amo2_nve_conservation.png', dpi=300)
    print("💾 Plot saved as 'amo2_nve_conservation.png'")

# FIXED: Standard pythonic main execution entry guard
if __name__ == "__main__":
    main()
