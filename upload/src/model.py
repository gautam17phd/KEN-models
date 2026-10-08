import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from torch_geometric.nn import global_add_pool

class CNNEdgeFeatureExtractor(nn.Module):
    """
    Accepts stacked physical matrices [Batch, Channels, N_atoms, N_atoms]
    and processes them into localized features via a 2D CNN framework.
    """
    def __init__(self, in_channels=3, out_features=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(32, out_features, kernel_size=1)
        )

    def forward(self, x):
        # Input shape: [Batch_Size, Channels, Max_Atoms, Max_Atoms]
        return self.cnn(x)


class RadialBasisFunctions(nn.Module):
    """
    Transforms raw edge distances into a smooth multi-channel Gaussian representation.
    Fixes 1D subtraction bugs via explicit 2D tensor broadcasting layout.
    """
    def __init__(self, num_rbf=32, d_max=7.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.register_buffer("centers", torch.linspace(0.0, d_max, num_rbf))
        self.gamma = 1.0 / ((d_max / num_rbf) ** 2)

    def forward(self, edge_len):
        # edge_len shape: [num_edges, 1]
        # self.centers.unsqueeze(0) shape: [1, num_rbf]
        # Output broadcasting shape: [num_edges, num_rbf]
        return torch.exp(-self.gamma * (edge_len - self.centers.unsqueeze(0)) ** 2)

class MACEWithCNNFeaturesMLIP(nn.Module):
    """
    Hybrid Machine Learning Interatomic Potential (MLIP).
    Combines 2D Spin Interaction CNN layers with 3D E(3)-Equivariant MACE Graph Blocks.
    """
    def __init__(self, num_elements=118, embedding_dim=64, lmax=2, num_rbf=32):
        super().__init__()

        # 1. Component Feature Extractors
        self.cnn_extractor = CNNEdgeFeatureExtractor(in_channels=3, out_features=16)
        self.radial_embedding = RadialBasisFunctions(num_rbf=num_rbf, d_max=6.0)

        # 2. Irreps Definition Specs
        cnn_dim = 16
        self.irreps_electronic = o3.Irreps(f"{embedding_dim + cnn_dim}x0e")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.irreps_single_message = o3.Irreps("32x0e + 16x1o + 8x2e")

        # 3. Geometric Edge Tensor Product Block
        self.edge_tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_sh,
            irreps_in2=self.irreps_electronic,
            irreps_out=self.irreps_single_message,
            shared_weights=False
        )

        # Radial Multi-Layer Perceptron (MLP) mapping RBF entries to Tensor Product Weights
        self.radial_net = nn.Sequential(
            nn.Linear(num_rbf, 128),
            nn.SiLU(),
            nn.Linear(128, self.edge_tp.weight_numel)
        )

        # 4. MACE High-Order Inter-Atomic Correlation Block
        self.irreps_correlations = o3.Irreps("64x0e + 32x1o + 16x2e")
        self.correlation_tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_single_message,
            irreps_in2=self.irreps_single_message,
            irreps_out=self.irreps_correlations,
            shared_weights=True
        )

        # 5. Base Chemical Identity Embeddings
        self.embedding = nn.Embedding(num_embeddings=num_elements, embedding_dim=embedding_dim)

        # Total Initial Node Allocation Input Specifications
        self.irreps_init_scalars = o3.Irreps(f"{embedding_dim + 16}x0e")
        self.irreps_node_features = (self.irreps_init_scalars + self.irreps_correlations).simplify()

        # Equivariant Mixing Projection Map Layer
        self.irreps_out_mix = o3.Irreps("128x0e + 32x1o")
        self.node_mix = Linear(
            irreps_in=self.irreps_node_features,
            irreps_out=self.irreps_out_mix
        )

        # 6. Decoupled Readout Property Heads
        self.energy_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        self.magmom_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch):
        pos = batch.pos
        atomic_numbers = batch.atomic_numbers
        edge_index = batch.edge_index
    
        # --------------------------------------------------------
        # PHASE 1: 2D CNN SELF-INTERACTION DIAGONAL MATRICES LOOKUP
        # --------------------------------------------------------
        cnn_features = self.cnn_extractor(batch.matrix_images)
    
        mol_idx = batch.matrix_mapping[:, 0]
        atom_idx = batch.matrix_mapping[:, 1]
    
        local_cnn_node_features = cnn_features[mol_idx, :, atom_idx, atom_idx]
    
        # --------------------------------------------------------
        # PHASE 2: CONSTRUCT INVARIANT INITIAL NODE SCALARS
        # --------------------------------------------------------
        node_embeddings = self.embedding(atomic_numbers)
        initial_scalars = torch.cat([node_embeddings, local_cnn_node_features], dim=-1)
    
        # --------------------------------------------------------
        # PHASE 3: FIXED FOR PBC - READ PRE-COMPUTED EDGE VECTORS
        # --------------------------------------------------------
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
    
        # REMOVED: edge_vec = pos[edge_dst] - pos[edge_src] <- Slow and breaks periodic physics
        # FIXED: Pulled directly from the fast dataloader cache
        edge_vec = batch.edge_vec 
        edge_len = torch.norm(edge_vec, dim=-1, keepdim=True) + 1e-8
    
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            edge_vec,
            normalize=True
        )
    
        rbf_features = self.radial_embedding(edge_len)
        weights = self.radial_net(rbf_features)
    
        source_electronic_features = initial_scalars
        edge_messages = self.edge_tp(
            sh,
            source_electronic_features[edge_src],
            weights
        )
    
        # --------------------------------------------------------
        # PHASE 4: MESSAGE POOLING & MANY-BODY CORRELATIONS
        # --------------------------------------------------------
        num_atoms = pos.size(0)
        A_messages = torch.zeros(num_atoms, self.irreps_single_message.dim, device=pos.device)
        A_messages.index_add_(0, edge_dst, edge_messages)
    
        B_correlations = self.correlation_tp(A_messages, A_messages)
    
        # --------------------------------------------------------
        # PHASE 5: FIXED FOR EQUIVARIANCE - ELIMINATE RAW INDEX SLICING
        # --------------------------------------------------------
        # FIXED: Directly concatenate full tensors instead of manual channel parsing
        combined_node_states = torch.cat([initial_scalars, B_correlations], dim=-1)
        mixed_node_states = self.node_mix(combined_node_states)
    
        # FIXED: Extract scalar channels using target layouts instead of hardcoded numbers
        dim_0e = self.irreps_out_mix[0].dim
        invariant_scalars = mixed_node_states[:, :dim_0e]
    
        # --------------------------------------------------------
        # PHASE 6: PROPERTY READOUTS
        # --------------------------------------------------------
        atomic_energies = self.energy_layer(invariant_scalars)
        pred_magmom = self.magmom_head(invariant_scalars)

        total_system_energy = global_add_pool(atomic_energies, batch.batch)
    
        return total_system_energy, pred_magmom


