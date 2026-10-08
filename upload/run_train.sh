#!/bin/bash
#SBATCH --account=nn4654k
#SBATCH --job-name=ken_train
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=train_output_%j.log

set -e

# --------------------------------------------------
# 1. Clean environment
# --------------------------------------------------
module purge

# Make sure old Conda Python cannot be picked up
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

# --------------------------------------------------
# 2. NVIDIA PyTorch ARM64 container
# --------------------------------------------------
CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif

# --------------------------------------------------
# 3. Make current project visible to Python
# --------------------------------------------------
export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "=========================================="
echo "🚀 Starting training job"
echo "=========================================="
echo "Date:         $(date)"
echo "Node:         $(hostname)"
echo "Architecture: $(uname -m)"
echo "Container:    ${CONTAINER}"

# --------------------------------------------------
# 4. Check GPU / PyTorch environment
# --------------------------------------------------
apptainer exec --nv "$CONTAINER" python -c "
import torch
import torch_geometric
import e3nn
import ase

print('------------------------------------------')
print('PyTorch:', torch.__version__)
print('PyG:', torch_geometric.__version__)
print('e3nn:', e3nn.__version__)
print('ASE:', ase.__version__)
print('CUDA:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())

print('------------------------------------------')
"

# --------------------------------------------------
# 5. Launch training inside the container
# --------------------------------------------------
echo "🚀 Launching training at $(date)"

apptainer exec --nv \
    "$CONTAINER" \
    python -m src.train

echo "🏁 Training finished at $(date)"
