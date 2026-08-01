"""
model.py
GNN (GAT + GraphNorm + Fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm, global_mean_pool

class GNNModel(nn.Module):
    def __init__(
        self,
        mol_input_dim=22,
        crystal_atom_dim=92,
        experimental_input_dim=6,
        global_feat_dim=18,
        mol_global_feat_dim=25,
        mol_hidden_dim=64,
        crystal_hidden_dim=64,
        experimental_hidden_dim=64,
        fusion_dim=64,
        combined_hidden_dim=256,
        feature_mask=None,
        interaction_mask=None
    ):
        super(GNNModel, self).__init__()

        if feature_mask is None:
            feature_mask = {
                "mol_graph": True,
                "mol_global": True,
                "crystal_graph": True,
                "crystal_global": True,
                "experimental": True
            }
        self.feature_mask = feature_mask

        if interaction_mask is None:
            interaction_mask = {
                "mol_exp": True,
                "crys_exp": True,
                "mol_crys": False,
                "triple": False
            }
        self.interaction_mask = interaction_mask

        # Molecular Graph GNN
        self.mol_conv1 = GATConv(mol_input_dim, mol_hidden_dim, heads=4, concat=True, edge_dim=7)
        self.mol_gn1 = GraphNorm(mol_hidden_dim * 4)
        self.mol_conv2 = GATConv(mol_hidden_dim * 4, mol_hidden_dim, heads=4, concat=True, edge_dim=7)
        self.mol_gn2 = GraphNorm(mol_hidden_dim * 4)
        self.mol_conv3 = GATConv(mol_hidden_dim * 4, mol_hidden_dim, heads=1, concat=False, edge_dim=7)
        self.mol_gn3 = GraphNorm(mol_hidden_dim)

        # Molecular Global Features
        self.mol_glb_fc1 = nn.Linear(mol_global_feat_dim, mol_hidden_dim)
        self.mol_glb_fc2 = nn.Linear(mol_hidden_dim, mol_hidden_dim)

        # Crystal Graph GNN
        crystal_edge_dim = 41
        self.crystal_conv1 = GATConv(crystal_atom_dim, crystal_hidden_dim, heads=4, concat=True, edge_dim=crystal_edge_dim)
        self.crystal_gn1 = GraphNorm(crystal_hidden_dim * 4)
        self.crystal_conv2 = GATConv(crystal_hidden_dim * 4, crystal_hidden_dim, heads=4, concat=True, edge_dim=crystal_edge_dim)
        self.crystal_gn2 = GraphNorm(crystal_hidden_dim * 4)
        self.crystal_conv3 = GATConv(crystal_hidden_dim * 4, crystal_hidden_dim, heads=1, concat=False, edge_dim=crystal_edge_dim)
        self.crystal_gn3 = GraphNorm(crystal_hidden_dim)

        # Crystal Global Features
        self.glb_feats_cryst1 = nn.Linear(global_feat_dim, crystal_hidden_dim)
        self.glb_feats_cryst2 = nn.Linear(crystal_hidden_dim, crystal_hidden_dim)

        # Experimental Branch
        self.experiment_fc1 = nn.Linear(experimental_input_dim, experimental_hidden_dim)
        self.experiment_fc2 = nn.Linear(experimental_hidden_dim, experimental_hidden_dim)

        # Projection layers
        self.mol_proj = nn.Linear(mol_hidden_dim, fusion_dim)
        self.crys_proj = nn.Linear(crystal_hidden_dim, fusion_dim)
        self.exp_proj = nn.Linear(experimental_hidden_dim, fusion_dim)

        # Calculate final fusion dimension
        total_hidden = 0
        if self.feature_mask["mol_graph"]: total_hidden += mol_hidden_dim
        if self.feature_mask["mol_global"]: total_hidden += mol_hidden_dim
        if self.feature_mask["crystal_graph"]: total_hidden += crystal_hidden_dim
        if self.feature_mask["crystal_global"]: total_hidden += crystal_hidden_dim
        if self.feature_mask["experimental"]: total_hidden += experimental_hidden_dim
        if self.interaction_mask["mol_exp"]: total_hidden += fusion_dim
        if self.interaction_mask["crys_exp"]: total_hidden += fusion_dim
        if self.interaction_mask["mol_crys"]: total_hidden += fusion_dim
        if self.interaction_mask["triple"]: total_hidden += fusion_dim
        print("Total fusion dimension:", total_hidden)

        self.fc1 = nn.Linear(total_hidden, combined_hidden_dim)
        self.fc2 = nn.Linear(combined_hidden_dim, combined_hidden_dim // 2)
        self.fc3 = nn.Linear(combined_hidden_dim // 2, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, mol_data, crystal_data, experimental_feat):
        features = []

        # Molecular Graph
        x_mol = None
        if self.feature_mask["mol_graph"]:
            x_mol = self.relu(self.mol_gn1(self.mol_conv1(mol_data.x, mol_data.edge_index, mol_data.edge_attr), mol_data.batch))
            x_mol = self.relu(self.mol_gn2(self.mol_conv2(x_mol, mol_data.edge_index, mol_data.edge_attr), mol_data.batch))
            x_mol = self.relu(self.mol_gn3(self.mol_conv3(x_mol, mol_data.edge_index, mol_data.edge_attr), mol_data.batch))
            x_mol = global_mean_pool(self.dropout(x_mol), mol_data.batch)
            features.append(x_mol)

        # Molecular Global
        mol_glb = None
        if self.feature_mask["mol_global"]:
            mol_glb = self.dropout(self.relu(self.mol_glb_fc2(self.relu(self.mol_glb_fc1(mol_data.mol_global_features)))))
            features.append(mol_glb)

        # Crystal Graph
        x_crystal = None
        if self.feature_mask["crystal_graph"]:
            x_crystal = self.relu(self.crystal_gn1(self.crystal_conv1(crystal_data.x, crystal_data.edge_index, crystal_data.edge_attr), crystal_data.batch))
            x_crystal = self.relu(self.crystal_gn2(self.crystal_conv2(x_crystal, crystal_data.edge_index, crystal_data.edge_attr), crystal_data.batch))
            x_crystal = self.relu(self.crystal_gn3(self.crystal_conv3(x_crystal, crystal_data.edge_index, crystal_data.edge_attr), crystal_data.batch))
            x_crystal = global_mean_pool(self.dropout(x_crystal), crystal_data.batch)
            features.append(x_crystal)

        # Crystal Global
        crystal_glb = None
        if self.feature_mask["crystal_global"]:
            crystal_glb = self.dropout(self.relu(self.glb_feats_cryst2(self.relu(self.glb_feats_cryst1(crystal_data.global_features)))))
            features.append(crystal_glb)

        # Experimental
        x_exp = None
        if self.feature_mask["experimental"]:
            x_exp = self.dropout(self.relu(self.experiment_fc2(self.relu(self.experiment_fc1(experimental_feat)))))
            features.append(x_exp)

        # Combine molecular representation
        mol_representation = None
        if x_mol is not None and mol_glb is not None:
            mol_representation = x_mol + mol_glb
        elif x_mol is not None:
            mol_representation = x_mol
        elif mol_glb is not None:
            mol_representation = mol_glb

        # Combine crystal representation
        crystal_representation = None
        if x_crystal is not None and crystal_glb is not None:
            crystal_representation = x_crystal + crystal_glb
        elif x_crystal is not None:
            crystal_representation = x_crystal
        elif crystal_glb is not None:
            crystal_representation = crystal_glb

        # Projections
        mol_p = self.mol_proj(mol_representation) if mol_representation is not None else None
        crys_p = self.crys_proj(crystal_representation) if crystal_representation is not None else None
        exp_p = self.exp_proj(x_exp) if x_exp is not None else None

        interaction_features = []
        if self.interaction_mask["mol_exp"] and mol_p is not None and exp_p is not None:
            interaction_features.append(mol_p * exp_p)
        if self.interaction_mask["crys_exp"] and crys_p is not None and exp_p is not None:
            interaction_features.append(crys_p * exp_p)
        if self.interaction_mask["mol_crys"] and mol_p is not None and crys_p is not None:
            interaction_features.append(mol_p * crys_p)
        if self.interaction_mask["triple"] and mol_p is not None and crys_p is not None and exp_p is not None:
            interaction_features.append(mol_p * crys_p * exp_p)

        combined = torch.cat(features + interaction_features, dim=1)
        out = self.dropout(self.relu(self.fc1(combined)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.fc3(out)
        return out, combined