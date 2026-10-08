#!/bin/bash
#SBATCH --account=<account name>
#SBATCH --job-name=pytorch_container_test
#SBATCH --partition=<partition name>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=pytorch_test_%j.log

CONTAINER=<container_path>/<filename>.sif

apptainer exec --nv "$CONTAINER" python -c "
import platform
import torch
import ase
import sklearn
import colorama
import matplotlib
import e3nn

print('--- VERIFICATION ---')
print('Architecture:', platform.machine())
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('ASE: OK')
print('scikit-learn: OK')
print('colorama: OK')
print('matplotlib: OK')
print('e3nn: OK')

try:
    import torch_geometric
    print('torch-geometric:', torch_geometric.__version__)
except ImportError:
    print('torch-geometric: MISSING')
"

apptainer exec --nv "$CONTAINER" \
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
