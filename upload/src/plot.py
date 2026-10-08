import os
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

def generate_training_summary_plots(energy_true_all, energy_pred_all, 
                                    force_true_all, force_pred_all, 
                                    magmom_true_all, magmom_pred_all, 
                                    history):
    """
    Generates a unified 4-panel visual dashboard displaying Energy, Force, 
    and Magnetic Moment parity plots along with dual-axis convergence curves.
    """
    fig = plt.figure(figsize=(18, 11))

    # ============================================================
    # SUBPLOT 1: ENERGY PARITY
    # ============================================================
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(energy_true_all, energy_pred_all, alpha=0.6, color='#1f77b4', s=15, edgecolors='none')
    e_min, e_max = min(energy_true_all), max(energy_true_all)
    elim = [e_min - 0.5, e_max + 0.5]
    ax1.plot(elim, elim, '--k', lw=2, alpha=0.7)
    ax1.set_xlim(elim)
    ax1.set_ylim(elim)
    ax1.set_xlabel("True Energy (eV/atom)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Predicted Energy (eV/atom)", fontsize=10, fontweight='bold')
    ax1.set_title("Energy Parity Plot", fontsize=12, fontweight='bold', color='#1f77b4')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ============================================================
    # SUBPLOT 2: FORCE PARITY (WITH SAFE SUBSAMPLING)
    # ============================================================
    #ax2 = fig.add_subplot(2, 3, 2)
    #ft, fp = np.array(force_true_all), np.array(force_pred_all)
    #sample_size = min(20000, len(ft))
    #f_idx = np.random.choice(len(ft), sample_size, replace=False)

    #ax2.scatter(ft[f_idx], fp[f_idx], alpha=0.15, color='#ff7f0e', s=3, edgecolors='none')
    #flim = [ft.min() - 0.2, ft.max() + 0.2]
    #ax2.plot(flim, flim, '--k', lw=2, alpha=0.7)
    #ax2.set_xlim(flim)
    #ax2.set_ylim(flim)
    #ax2.set_xlabel("True Force (eV/Å)", fontsize=10, fontweight='bold')
    #ax2.set_ylabel("Predicted Force (eV/Å)", fontsize=10, fontweight='bold')
    #ax2.set_title("Force Parity Plot (Subsampled)", fontsize=12, fontweight='bold', color='#ff7f0e')
    #ax2.grid(True, linestyle=':', alpha=0.6)

    ax2 = fig.add_subplot(2, 3, 2)
    
    # 1. 🌟 FLATTEN: Safely flatten all multidimensional atomic matrices into pure 1D components
    ft = np.concatenate([np.atleast_1d(f).ravel() for f in force_true_all])
    fp = np.concatenate([np.atleast_1d(f).ravel() for f in force_pred_all])

    # 2. 🌟 PLOT ALL: Remove random selection index entirely and pass full data arrays
    ax2.scatter(ft, fp, alpha=0.15, color='#ff7f0e', s=3, edgecolors='none')
    
    # 3. 🌟 PERCENTILE BOUNDS: Protects the plot frame scale from expanding due to rare high-energy spikes
    f_min = np.percentile(ft, 1) - 0.5
    f_max = np.percentile(ft, 99) + 0.5
    flim = [f_min, f_max]
    
    # 4. Finalize visual frame borders and grid overlays
    ax2.plot(flim, flim, '--k', lw=2, alpha=0.7)
    ax2.set_xlim(flim)
    ax2.set_ylim(flim)
    ax2.set_xlabel("True Force (eV/Å)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Predicted Force (eV/Å)", fontsize=10, fontweight='bold')
    ax2.set_title("Force Parity Plot (All Components)", fontsize=12, fontweight='bold', color='#ff7f0e')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # ============================================================
    # SUBPLOT 3: MAGNETIC MOMENTS PARITY 
    # ============================================================
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(magmom_true_all, magmom_pred_all, alpha=0.5, color='#2ca02c', s=15, edgecolors='none')
    m_min, m_max = min(magmom_true_all), max(magmom_true_all)
    if m_min == m_max:
        mlim = [m_min - 1.0, m_max + 1.0]
    else:
        mlim = [m_min - 0.2, m_max + 0.2]
    ax3.plot(mlim, mlim, '--k', lw=1.5, alpha=0.7)
    ax3.set_xlim(mlim)
    ax3.set_ylim(mlim)
    ax3.set_xlabel("True Magmom", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Predicted Magmom", fontsize=10, fontweight='bold')
    ax3.set_title("Magnetic Moment Parity Plot", fontsize=12, fontweight='bold', color='#2ca02c')
    ax3.grid(True, linestyle=':', alpha=0.6)

    # ============================================================
    # SUBPLOT 4: TRAINING CURVES (DUAL TWIN-Y SCALED AXES)
    # ============================================================
    ax4_energy = fig.add_subplot(2, 1, 2)  
    ax4_force = ax4_energy.twinx()        

    line1 = ax4_energy.plot(history["energy_mae"], '-', lw=2.5, color='#1f77b4', label="Energy MAE (meV/atom)")
    line2 = ax4_force.plot(history["force_mae"], '-', lw=2.5, color='#ff7f0e', label="Force MAE (meV/Å)")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4_energy.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    ax4_energy.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax4_energy.set_ylabel("Energy MAE [meV/atom]", color='#1f77b4', fontsize=11, fontweight='bold')
    ax4_energy.tick_params(axis='y', labelcolor='#1f77b4')

    ax4_force.set_ylabel("Force MAE [meV/Å]", color='#ff7f0e', fontsize=11, fontweight='bold')
    ax4_force.tick_params(axis='y', labelcolor='#ff7f0e')

    ax4_energy.set_title("Multi-Task Training Loss Convergence History", fontsize=13, fontweight='bold', pad=12)
    ax4_energy.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # Optional: Save a static image copy directly to your models/ folder
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/training_summary_dashboard.png", dpi=300)
    print(Fore.GREEN + "🖼️  Saved publication quality summary dashboard to models/training_summary_dashboard.png")
    print(Fore.GREEN + Style.BRIGHT + "\n✅ All Multi-Task Train Metrics Plotted Successfully with Dual Y-Axis Scaling!")

def testing_summary_plots(true_energies, predicted_energies, true_forces, predicted_forces, 
                          true_magmoms, predicted_magmoms, history):


    fig, axs = plt.subplots(2, 2, figsize=(15, 14))
    
    axs = axs.flatten()
    
    # ------------------------------------------------------------
    # 1. ENERGY PARITY PLOT (Top-Left, Index 0)
    # ------------------------------------------------------------
    axs[0].scatter(true_energies, predicted_energies, alpha=0.6, color='#1f77b4', edgecolors='none', s=20)
    min_e = min(true_energies.min(), predicted_energies.min())
    max_e = max(true_energies.max(), predicted_energies.max())
    elim = [min_e - 0.1, max_e + 0.1]
    
    axs[0].plot(elim, elim, '--k', alpha=0.7, lw=1.5)
    axs[0].set_xlim(elim)
    axs[0].set_ylim(elim)
    axs[0].set_xlabel("True Energy (eV/atom)", fontweight='bold', fontsize=10)
    axs[0].set_ylabel("Predicted Energy (eV/atom)", fontweight='bold', fontsize=10)
    axs[0].set_title("Absolute Energy Per Atom Parity Plot", fontweight='bold', color='#1f77b4', fontsize=12)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    # ------------------------------------------------------------
    # 2. FORCE VECTOR PARITY PLOT (Top-Right, Index 1)
    # ------------------------------------------------------------

    # 1. 🌟 ALL COMPONENTS: Keep your flat arrays exactly as they are
    ft_flat, fp_flat = true_forces.flatten(), predicted_forces.flatten()

    # 2. 🌟 PLOT ALL: Pass the entire unsampled flat arrays directly into the scatter function
    axs[1].scatter(ft_flat, fp_flat, alpha=0.15, color='#ff7f0e', edgecolors='none', s=4)
    
    # 3. 🌟 PERCENTILE BOUNDS: Zooms past rare high-energy spikes to frame the main dataset distribution
    f_min = np.percentile(ft_flat, 1) - 0.5
    f_max = np.percentile(ft_flat, 99) + 0.5
    flim = [f_min, f_max]

    # 4. Finalize visual frame layout borders and grid overlays
    axs[1].plot(flim, flim, '--k', alpha=0.7, lw=1.5)
    axs[1].set_xlim(flim)
    axs[1].set_ylim(flim)
    axs[1].set_xlabel("True Force Vector Components (eV/Å)", fontweight='bold', fontsize=10)
    axs[1].set_ylabel("Predicted Force Vector Components (eV/Å)", fontweight='bold', fontsize=10)
    axs[1].set_title("Force Parity Plot (All Components)", fontweight='bold', color='#ff7f0e', fontsize=12)
    axs[1].grid(True, linestyle=':', alpha=0.6)


    # ------------------------------------------------------------
    # 3. FORCE ERROR COMPONENT RESIDUAL DISTRIBUTION (Bottom-Left, Index 2)
    # ------------------------------------------------------------
    force_residuals = fp_flat - ft_flat
    axs[2].hist(force_residuals, bins=100, color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.4)
    axs[2].set_yscale('log')  # Logarithmic scale isolates long-tail error spaces cleanly
    
    axs[2].set_xlabel("Force Error Component Residuals (eV/Å)", fontweight='bold', fontsize=10)
    axs[2].set_ylabel("Observation Count (Log Scale)", fontweight='bold', fontsize=10)
    axs[2].set_title("Authentic Unpadded Force Component Error Profile", fontweight='bold', color='#d62728', fontsize=12)
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    # ------------------------------------------------------------
    # 4. MAGNETIC MOMENT PARITY PLOT (Bottom-Right, Index 3)
    # ------------------------------------------------------------
    axs[3].scatter(true_magmoms, predicted_magmoms, alpha=0.5, color='#2ca02c', edgecolors='none', s=20)
    min_m, max_m = true_magmoms.min(), true_magmoms.max()
    mlim = [min_m - 0.5, max_m + 0.5] if min_m == max_m else [min_m - 0.1, max_m + 0.1]
    
    axs[3].plot(mlim, mlim, '--k', alpha=0.7, lw=1.5)
    axs[3].set_xlim(mlim)
    axs[3].set_ylim(mlim)
    axs[3].set_xlabel("True Magnetic Moments (μB)", fontweight='bold', fontsize=10)
    axs[3].set_ylabel("Predicted Magnetic Moments (μB)", fontweight='bold', fontsize=10)
    axs[3].set_title("Magnetic Moment Invariant Parity Plot", fontweight='bold', color='#2ca02c', fontsize=12)
    axs[3].grid(True, linestyle=':', alpha=0.6)
    
    # Tighten layout constraints to prevent text labels overlapping between panels
    plt.tight_layout()
    fig.savefig("plots/testing_summary_dashboard.png", dpi=300)
    print(Fore.GREEN + Style.BRIGHT + "✅ Testing Analytics Panel Matrix Rendered Successfully!")
