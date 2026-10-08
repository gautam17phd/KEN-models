import numpy as np
import os
import torch
import warnings
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from colorama import Fore, Style, init
from sklearn.metrics import mean_absolute_error, mean_squared_error
init(autoreset=True)
from src.plot import generate_training_summary_plots, testing_summary_plots
from src.dataset import HE26MolecularDataset, hybrid_collate_fn
from src.model import MACEWithCNNFeaturesMLIP

def save_mlip_checkpoint(model, optimizer, epoch, val_loss, hyperparams, filepath="best_mlip_model.pt"):
    """
    Saves the complete state of the Hybrid MACE-CNN model and training environment.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        # Save structural parameters to prevent layout mismatch errors on reload
        'hyperparams': hyperparams
    }

    # Save safely using a temporary file to avoid corruption during disk writes
    tmp_filepath = filepath + ".tmp"
    torch.save(checkpoint, tmp_filepath)
    os.replace(tmp_filepath, filepath)
    print(f"--- Checkpoint safely saved to '{filepath}' at epoch {epoch} ---")

def resume_mlip_checkpoint(filepath, device=torch.device('cpu')):
    """
    Loads a saved checkpoint, rebuilds the exact architecture matching
    the saved configuration, and restores training states.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at path: {filepath}")

    # Map storage directly to the target training device
    checkpoint = torch.load(filepath, map_location=device)

    # 1. Rebuild the exact model structure using saved hyperparameters
    hparams = checkpoint['hyperparams']
    model = MACEWithCNNFeaturesMLIP(
        num_elements=hparams.get('num_elements', 118),
        embedding_dim=hparams.get('embedding_dim', 64),
        lmax=hparams.get('lmax', 2),
        num_rbf=hparams.get('num_rbf', 32)
    )

    # 2. Load weights and optimizer parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"--- Checkpoint loaded. Resuming from Epoch {checkpoint['epoch']} (Saved Loss: {checkpoint['val_loss']:.4f}) ---")
    return model, checkpoint['optimizer_state_dict'], checkpoint['epoch'], checkpoint['val_loss']
    
def train_step(model, batch, optimizer, device):
    """
    Trains the MACEWithCNNFeaturesMLIP model on a single batch,
    computing analytical forces safely across periodic multi-molecule cells.
    """
    model.train()
    optimizer.zero_grad()

    # Enable gradients on absolute positions before any forward calculations
    batch.pos = batch.pos.clone().detach().requires_grad_(True)

    # =====================================================================
    # FIXED FOR PERIODIC GRADIENTS: RE-ATTACH SPATIAL BASE GRAPH
    # =====================================================================
    # Traces gradients (dE/dpos) natively while keeping periodic shifts intact.
    edge_src = batch.edge_index[0]
    edge_dst = batch.edge_index[1]
    
    # Calculate relative open-space displacement
    dr_open = batch.pos[edge_dst] - batch.pos[edge_src]
    
    # Isolate cell offsets calculated by ASE
    with torch.no_grad():
        cell_offsets = batch.edge_vec - dr_open
        
    # Re-inject the dynamic position tensor back into your boundary coordinates
    batch.edge_vec = dr_open + cell_offsets

    # Execute Forward Pass
    pred_energy, pred_magmom = model(batch)

    pred_energy_flat = pred_energy.view(-1)
    true_energy = batch.y.view(-1)

    # Analytical force extraction via vector-Jacobian product contraction
    raw_gradients = torch.autograd.grad(
        outputs=pred_energy_flat,
        inputs=batch.pos,
        grad_outputs=torch.ones_like(pred_energy_flat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    pred_forces = -raw_gradients

    # =====================================================================
    # DYNAMIC ENERGY/ATOM NORMALIZATION & MASKED READING PROPERTIES
    # =====================================================================
    atoms_per_graph = torch.bincount(batch.batch).float()
    pred_energy_per_atom = pred_energy_flat / atoms_per_graph
    true_energy_per_atom = true_energy / atoms_per_graph

    # 1. Normalized Energy Loss
    energy_loss = torch.nn.functional.mse_loss(
        pred_energy_per_atom,
        true_energy_per_atom
    )

    # 2. Masked Local Force Loss (Excludes padding layers)
    real_atoms_mask = (batch.atomic_numbers > 0)
    force_loss = torch.nn.functional.mse_loss(
        pred_forces[real_atoms_mask],
        batch.forces[real_atoms_mask]
    )

    # 3. FIXED: Masked Magnetic Moment Loss (Prevents learning dummy parameters)
    magmom_loss = torch.nn.functional.mse_loss(
        pred_magmom.view(-1)[real_atoms_mask],
        batch.magmom.view(-1)[real_atoms_mask]
    )

    # Multi-task weight balancing strategy
    total_loss = 1.0 * energy_loss + 100.0 * force_loss + 10.0 * magmom_loss

    total_loss.backward()

    # Prevent gradient explosion across deep e3nn layouts
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    return (
        total_loss.item(),
        energy_loss.item(),
        force_loss.item(),
        magmom_loss.item(),
        
        pred_energy_flat.detach().cpu(),
        true_energy.detach().cpu(),
        
        pred_forces.detach().cpu(),
        batch.forces.detach().cpu(),
        
        pred_magmom.detach().cpu(),
        batch.magmom.detach().cpu()
    )

if __name__ == "__main__":
    # Safe multi-processing context initialization mandatory for Mac execution
    #torch.multiprocessing.set_start_method('spawn', force=True)

    # 1. Load the dataset (no start method tweaks needed)
    dataset = HE26MolecularDataset(pickle_path='data/training.pkl')

    # 2. Rock-solid single-process DataLoader configuration for HPC
    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=hybrid_collate_fn,
        num_workers=0,            # Run directly on the main thread (No multiprocessing crashes!)
        persistent_workers=False,  # Must be False when num_workers=0
        pin_memory=True           # Keep True! This streams your molecules directly into GPU RAM
    )

    #dataset = HE26MolecularDataset(pickle_path='data/training.pkl')
   
    # 1. Detect how many CPUs Slurm assigned to this task (defaults to 4 if local testing)
    #cluster_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    # 2. Optimal DataLoader configuration for HPC
    #train_loader = DataLoader(
     #   dataset,
     #   batch_size=16,
     #   shuffle=True,
     #   collate_fn=hybrid_collate_fn,
     #   num_workers=cluster_workers,
     #   persistent_workers=True,  # Keeps data loader processes alive between epochs
     #   pin_memory=True           # Speeds up transferring graphs from CPU RAM to GPU memory
    #)

    # ============================================================
    # MODEL SETUP & OPTIMIZER WORKFLOW
    # ============================================================
    hyperparams = {
        'num_elements': 118,
        'embedding_dim': 64,   # 👈 UPGRADED: Expanded feature channels
        'lmax': 2,
        'num_rbf': 32          # 👈 UPGRADED: Finer distance grids
    }
    # FIXED: Detect and assign Apple Silicon GPU backend natively
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"🍏 Targeting Device Acceleration Context: {device}")
    checkpoint_path = "model/best_ken_model.pt"
    
    # Initialize Workflow (Using your external resume function)
    if os.path.exists(checkpoint_path):
        model, saved_opt_state, start_epoch, _ = resume_mlip_checkpoint(checkpoint_path, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(saved_opt_state)
        start_epoch += 1 
        best_physical_score = float('inf') 
    else:
        model = MACEWithCNNFeaturesMLIP(**hyperparams).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        start_epoch = 0
        best_physical_score = float('inf') 
    
    
    # ============================================================
    # SCHEDULER SETUP
    # ============================================================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )
    
    print(f"📖 Loaded {len(dataset)} molecular systems from training.pkl.")
    print(Fore.CYAN + Style.BRIGHT + f"\n🚀 Starting Hybrid MACE + CNN MLIP Training on device: {device}\n")
    
    # ============================================================
    # TRAINING CONFIG
    # ============================================================
    NUM_EPOCHS = 60
    
    history = {
        "loss": [],
        "energy_mae": [],  
        "energy_rmse": [],
        "force_mae": [],
        "force_rmse": [],
        "magmom_mae": [],
        "magmom_rmse": [],
    }
    
    # ============================================================
    # MAIN TRAINING LOOP
    # ============================================================
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_energy = 0.0
        epoch_force = 0.0
        epoch_magmom = 0.0
        steps = 0
    
        energy_true_all = []
        energy_pred_all = []
        force_true_all = []
        force_pred_all = []
        magmom_true_all = []
        magmom_pred_all = []
        atoms_count_all = []
    
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
    
            (
                total_l, energy_l, force_l, magmom_l,
                pred_e, true_e,
                pred_f, true_f,
                pred_m, true_m
            ) = train_step(model, batch, optimizer, device)
    
            epoch_loss += total_l
            epoch_energy += energy_l
            epoch_force += force_l
            epoch_magmom += magmom_l
            steps += 1
    
            # Move raw prediction vectors to CPU NumPy arrays
            pred_e_np = pred_e.detach().cpu().numpy().reshape(-1)
            true_e_np = true_e.detach().cpu().numpy().reshape(-1)
    
            # 1. Map atomic references dictionary into lookup array
            max_z = max(dataset.atomic_refs.keys())
            ref_lookup = np.zeros(max_z + 1, dtype=np.float32)
            for z, ref_val in dataset.atomic_refs.items():
                ref_lookup[z] = ref_val
    
            # 2. Extract PyG flattened atomic records and batch structural boundaries
            batch_z_np = batch.atomic_numbers.detach().cpu().numpy().astype(np.int32)
            batch_indices = batch.batch.detach().cpu().numpy().astype(np.int32)
            num_graphs = len(pred_e_np)
    
            # 3. Calculate atom counts per structure for per-atom metrics
            graphs_atoms_counts = np.bincount(batch_indices, minlength=num_graphs)
            atoms_count_all.extend(graphs_atoms_counts)
    
            # 4. Map the reference energy value to every single atom in the batch
            atom_ref_energies = ref_lookup[batch_z_np]
            
            # 5. Sum the atomic reference energies grouped by their molecule index
            batch_baselines = np.bincount(
                batch_indices, 
                weights=atom_ref_energies, 
                minlength=num_graphs
            ).astype(np.float32)
    
            # 6. Reconstruct absolute system energies for the Parity plots
            pred_e_real = pred_e_np + batch_baselines
            true_e_real = true_e_np + batch_baselines
    
            pred_f_real = pred_f.detach().cpu().numpy()
            true_f_real = true_f.detach().cpu().numpy()
    
            # Storage extension
            energy_true_all.extend(true_e_real)
            energy_pred_all.extend(pred_e_real)
            force_true_all.extend(true_f_real.reshape(-1))
            force_pred_all.extend(pred_f_real.reshape(-1))
            magmom_true_all.extend(true_m.detach().cpu().numpy().reshape(-1))
            magmom_pred_all.extend(pred_m.detach().cpu().numpy().reshape(-1))
    
        # ========================================================
        # METRICS EVALUATION WITH SYSTEM-SIZE NORMALIZATION
        # ========================================================
        atoms_count_all = np.array(atoms_count_all, dtype=np.float32)
        
        # Calculate Per-Atom Energy deviations to remove system size bias
        energy_true_per_atom = np.array(energy_true_all) / atoms_count_all
        energy_pred_per_atom = np.array(energy_pred_all) / atoms_count_all
    
        # Track absolute energy metrics per atom to clear the 750 meV bottleneck
        energy_mae_mev_per_atom = mean_absolute_error(energy_true_per_atom, energy_pred_per_atom) * 1000
        energy_rmse_mev_per_atom = np.sqrt(mean_squared_error(energy_true_per_atom, energy_pred_per_atom)) * 1000
    
        force_mae_mev = mean_absolute_error(force_true_all, force_pred_all) * 1000
        force_rmse_mev = np.sqrt(mean_squared_error(force_true_all, force_pred_all)) * 1000
    
        magmom_mae = mean_absolute_error(magmom_true_all, magmom_pred_all)
        magmom_rmse = np.sqrt(mean_squared_error(magmom_true_all, magmom_pred_all))
    
        # Save tracking history
        history["loss"].append(epoch_loss / steps)
        history["energy_mae"].append(energy_mae_mev_per_atom)     
        history["energy_rmse"].append(energy_rmse_mev_per_atom)   
        history["force_mae"].append(force_mae_mev)
        history["force_rmse"].append(force_rmse_mev)
        history["magmom_mae"].append(magmom_mae)
        history["magmom_rmse"].append(magmom_rmse)
    
        # ========================================================
        # SCHEDULER UPDATE & STEP TRACKING
        # ========================================================
        current_epoch_loss = epoch_loss / steps
        scheduler.step(current_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
    
        # Colorful console output
        print(Fore.YELLOW + Style.BRIGHT + f"\n══════════ Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} ══════════")
        print(Fore.GREEN + f"📉 Total Loss       : {current_epoch_loss:.6f}")
        print(Fore.LIGHTBLACK_EX + f"⚙️ Active Learn Rate: {current_lr:.6e}")
        print(Fore.CYAN + f"⚡ Energy MSE       : {epoch_energy/steps:.6f}")
        print(Fore.MAGENTA + f"🧲 Force MSE        : {epoch_force/steps:.6f}")
        print(Fore.BLUE + f"📏 Energy MAE       : {energy_mae_mev_per_atom:.2f} meV/atom")
        print(Fore.BLUE + f"📐 Energy RMSE      : {energy_rmse_mev_per_atom:.2f} meV/atom")
        print(Fore.RED + f"💪 Force MAE        : {force_mae_mev:.2f} meV/Å")
        print(Fore.RED + f"🏋️ Force RMSE       : {force_rmse_mev:.2f} meV/Å")
        print(Fore.WHITE + f"🧲 Magmom MAE       : {magmom_mae:.4f} μB")
    
        # ========================================================
        # BULLETPROOF CHECKPOINT TRIGGER (Combined Physical Score)
        # ========================================================
        # Combines meV/atom and meV/Å directly to balance structural and thermodynamic health
        current_physical_score = energy_mae_mev_per_atom + force_mae_mev
    
        if current_physical_score < best_physical_score:
            best_physical_score = current_physical_score  
            save_mlip_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=best_physical_score,
                hyperparams=hyperparams,
                filepath=checkpoint_path
            )

    print(Fore.CYAN + "\n📊   Training complete. Generating performance diagnostic visuals...")
    energy_true_per_atom = np.array(energy_true_all) / atoms_count_all
    energy_pred_per_atom = np.array(energy_pred_all) / atoms_count_all
    
    generate_training_summary_plots(
        energy_true_per_atom, energy_pred_per_atom,
        force_true_all, force_pred_all,
        magmom_true_all, magmom_pred_all,
        history
    )
    
    # ==========================================
    # TESTING / EVALUATION PIPELINE
    # ==========================================
    test_dataset = HE26MolecularDataset(pickle_path='data/testing.pkl')
    
    # Safely select macOS workers
    #num_test_workers = 2 if not torch.cuda.is_available() else 4
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=hybrid_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"🧪 Loaded {len(test_dataset)} testing systems.")
    
    # Pre-build reference lookup array from the training baseline step
    max_z = max(test_dataset.atomic_refs.keys())
    ref_lookup = np.zeros(max_z + 1, dtype=np.float32)
    for z, ref_val in test_dataset.atomic_refs.items():
        ref_lookup[z] = ref_val
    
    # FIXED: Switch model to eval mode, but DO NOT freeze weights with requires_grad_(False).
    # Analytical force calculations MUST trace gradients back through the layers to pos.
    model.eval()
    
    true_energies = []
    predicted_energies = []
    true_forces = []
    predicted_forces = []
    true_magmoms = []
    predicted_magmoms = []
    
    # FIXED: Use inference_mode/no_grad context for standard passes, but we handle it manually
    # to let autograd evaluate the position components explicitly.
    for batch in test_loader:
        batch = batch.to(device)
        
        # Enable tracking explicitly on the input coordinates
        batch.pos.requires_grad_(True)
        
        # =====================================================================
        # FIXED FOR PERIODIC GRADIENTS: RE-MAP SPATIAL BASIS TO TRACK batch.pos
        # =====================================================================
        # Because batch.edge_vec was pre-computed on the CPU, it is a static array.
        # To get analytical forces (dE/dpos), we must add back the spatial position graph 
        # shifts while preserving your periodic image lattice modifications.
        edge_src = batch.edge_index[0]
        edge_dst = batch.edge_index[1]
        
        # Calculate the dynamic displacement out-of-cell vector
        dr_open = batch.pos[edge_dst] - batch.pos[edge_src]
        
        # Isolate the exact wrapping offsets computed by ASE
        with torch.no_grad():
            cell_offsets = batch.edge_vec - dr_open
            
        # Re-attach the dynamic pos gradient graph to your periodic boundary vectors
        batch.edge_vec = dr_open + cell_offsets 
        
        # Execute Forward Pass
        pred_energy_force, pred_magmom_force = model(batch)
        pred_energy_flat = pred_energy_force.view(-1)
        
        # Calculate forces: F = -dE/dpos
        raw_gradients = torch.autograd.grad(
            outputs=pred_energy_flat,
            inputs=batch.pos,
            grad_outputs=torch.ones_like(pred_energy_flat),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )
        
        pred_force = -raw_gradients[0] 
    
        # Extract dynamic padding masks to isolate dummy elements
        real_mask_cpu = (batch.atomic_numbers > 0).cpu().numpy()
        batch_z_np = batch.atomic_numbers.cpu().numpy().astype(np.int32)
        batch_indices = batch.batch.cpu().numpy().astype(np.int32)
        num_graphs = len(pred_energy_flat)
    
        # Store Masked Local Property Values
        true_forces.extend(batch.forces.cpu().numpy()[real_mask_cpu])
        predicted_forces.extend(pred_force.detach().cpu().numpy()[real_mask_cpu])
        
        true_magmoms.extend(batch.magmom.cpu().numpy().reshape(-1)[real_mask_cpu])
        predicted_magmoms.extend(pred_magmom_force.detach().cpu().numpy().reshape(-1)[real_mask_cpu])
    
        # Reconstruct system scale tracking arrays
        graphs_atoms_counts = np.bincount(batch_indices[real_mask_cpu], minlength=num_graphs)
        
        atom_ref_energies = ref_lookup[batch_z_np]
        batch_baselines = np.bincount(
            batch_indices, 
            weights=atom_ref_energies, 
            minlength=num_graphs
        ).astype(np.float32)
    
        # Reconstruct true absolute energy configurations
        pred_e_real = pred_energy_flat.detach().cpu().numpy() + batch_baselines
        
        # FIXED: batch.y is the clean, shifted delta value from your __getitem__.
        # Adding batch_baselines to it directly extracts the raw target row['energy'].
        true_e_real = batch.y.view(-1).cpu().numpy() + batch_baselines
    
        # Normalize by valid unpadded system scale factor
        pred_e_per_atom = pred_e_real / graphs_atoms_counts
        true_e_per_atom = true_e_real / graphs_atoms_counts
    
        true_energies.extend(true_e_per_atom)
        predicted_energies.extend(pred_e_per_atom)
        
    # ==========================================
    # FINAL VECTOR MATRIX GENERATION
    # ==========================================
    true_energies = np.array(true_energies)
    predicted_energies = np.array(predicted_energies)
    true_forces = np.array(true_forces)
    predicted_forces = np.array(predicted_forces)
    true_magmoms = np.array(true_magmoms)
    predicted_magmoms = np.array(predicted_magmoms)
    
    # Compute True Errors
    energy_mae = np.mean(np.abs(predicted_energies - true_energies))
    energy_rmse = np.sqrt(np.mean((predicted_energies - true_energies)**2))
    force_mae = np.mean(np.abs(predicted_forces - true_forces))
    force_rmse = np.sqrt(np.mean((predicted_forces - true_forces)**2))
    magmom_mae = np.mean(np.abs(predicted_magmoms - true_magmoms))
    magmom_rmse = np.sqrt(np.mean((predicted_magmoms - true_magmoms)**2))
    
    print("\n📊 FINAL TEST RESULTS (UNPADDED & RE-BASELINED)")
    print("=" * 50)
    print(f"ENERGY MAE (per atom)  : {energy_mae:.6f} eV/atom")
    print(f"ENERGY RMSE (per atom) : {energy_rmse:.6f} eV/atom")
    print(f"\nFORCE MAE   : {force_mae:.6f} eV/Å")
    print(f"FORCE RMSE  : {force_rmse:.6f} eV/Å")
    print(f"\nMAGMOM MAE   : {magmom_mae:.6f} μB")
    print(f"MAGMOM RMSE  : {magmom_rmse:.6f} μB")
    
    testing_summary_plots(true_energies, predicted_energies, 
                          true_forces, predicted_forces, 
                          true_magmoms, predicted_magmoms,
                          history)


    # 1. Load your model structure + weights normally
    device = torch.device("cpu")
    model, _, _, _ = resume_mlip_checkpoint("model/best_ken_model.pt", device=device)
    model.eval()
    
    # 2. Define the Wrapper using native PyG object assembly
    class TorchScriptModelWrapper(torch.nn.Module):
        def __init__(self, trained_model):
            super().__init__()
            self.model = trained_model
    
        def forward(self, pos, atomic_numbers, edge_index, edge_vec, matrix_images, matrix_mapping, batch_indices):
            """
            Accepts the raw structural tensors, builds a unified PyG graph container 
            on the fly, and runs the downstream model calculations smoothly.
            """
            # Assemble individual properties into a clean Graph object container
            g = Data(
                pos=pos,
                atomic_numbers=atomic_numbers,
                edge_index=edge_index,
                edge_vec=edge_vec,  # 👈 FIXED: Injected required periodic vector path
                batch=batch_indices,
                matrix_mapping=matrix_mapping
            )
            
            # Turn the single graph dictionary structure into a global batch
            batch_obj = Batch.from_data_list([g])
            batch_obj.matrix_images = matrix_images
            
            return self.model(batch_obj)
    
    # 3. Create raw tensors representing a system with 10 atoms and 30 edges
    num_atoms = 10
    num_edges = 30
    
    mock_pos = torch.randn(num_atoms, 3)               
    mock_atomic_numbers = torch.randint(1, 118, (num_atoms,))
    mock_edge_index = torch.randint(0, num_atoms, (2, num_edges))
    mock_edge_vec = torch.randn(num_edges, 3)  # 👈 FIXED: Added mock periodic vectors array
    mock_matrix_images = torch.randn(1, 3, num_atoms, num_atoms) 
    mock_matrix_mapping = torch.stack([torch.zeros(num_atoms), torch.arange(num_atoms)], dim=-1).long()
    mock_batch_indices = torch.zeros(num_atoms, dtype=torch.long) 
    
    # Group into the signature arguments tuple matching your compiled signature layout
    example_inputs = (
        mock_pos, 
        mock_atomic_numbers, 
        mock_edge_index, 
        mock_edge_vec,         # New Argument in position 4
        mock_matrix_images,    # Moves to position 5
        mock_matrix_mapping,   # Moves to position 6
        mock_batch_indices     # Moves to position 7
    )
    
    # 4. Run the standard JIT Trace
    try:
        print("⏳ Tracing network compilation tracks...")
        wrapper = TorchScriptModelWrapper(model)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            # strict=False lets e3nn's internal checking loops pass without warnings
            traced_compiled_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        
        # 5. Save the final standalone asset
        traced_compiled_model.save("../ken/KENJI.ptc")
        print("🎉 Success! 'KENJI.ptc' has been successfully compiled and saved.")
        print("Your model is exported and ready for production deployment.")
    
    except Exception as e:
        print(f"❌ Tracing failed: {e}")




        
    

 

