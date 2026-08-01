"""
molecule_features.py

"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, AllChem, rdMolDescriptors
from torch_geometric.data import Data
import random, numpy as np, torch
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ==================== atom feature  ====================
def get_atom_features(mol):
    try:      
        atom_features = []
        for atom in mol.GetAtoms():    
            features = [
                atom.GetAtomicNum(),
                atom.GetMass(),
                atom.GetDegree(),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                atom.GetTotalValence(),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs()
            ]
            formal_charge = [0]*7
            charge = atom.GetFormalCharge()
            charge_map = [-3, -2, -1, 0, 1, 2, 3]
            if charge in charge_map:
                formal_charge[charge_map.index(charge)] = 1
            hybridization = [0]*7
            hybrid = atom.GetHybridization()
            hybrid_map = {
                Chem.rdchem.HybridizationType.S:0,
                Chem.rdchem.HybridizationType.SP:1,
                Chem.rdchem.HybridizationType.SP2:2,
                Chem.rdchem.HybridizationType.SP3:3,
                Chem.rdchem.HybridizationType.SP2D:4,
                Chem.rdchem.HybridizationType.SP3D:5,
                Chem.rdchem.HybridizationType.SP3D2:6
            }
            if hybrid in hybrid_map:
                hybridization[hybrid_map[hybrid]] = 1
            atom_features.append(features + formal_charge + hybridization)
    except Exception as e:
        print(f"Error in getting atom features: {e}")
    return atom_features

def compute_bond_length(mol):
    try:
        conf = mol.GetConformer()
        bond_lengths = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pos_i = np.array(conf.GetAtomPosition(i))
            pos_j = np.array(conf.GetAtomPosition(j))
            bond_length = np.linalg.norm(pos_i - pos_j)
            bond_lengths.append(bond_length)
    except Exception as e:
        print(f' error in computing bond length: {e}')
    return bond_lengths

def get_bond_features(bond, bond_length):
    try:
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_type_feat = [1, 0, 0]
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_type_feat = [0, 1, 0]
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_type_feat = [0, 0, 1]
        else:
            bond_type_feat = [0, 0, 0]
        stereo = bond.GetStereo()
        stereo_feat = [1, 0] if stereo == Chem.rdchem.BondStereo.STEREOZ else [0, 1] if stereo == Chem.rdchem.BondStereo.STEREOE else [0, 0] 
        conjugation_feat = [1] if bond.GetIsConjugated() else [0]
        bond_features = bond_type_feat + stereo_feat + conjugation_feat + [bond_length]
        return bond_features
    except Exception as e:
        print(f'Error in getting bond features : {e}')

# ==================== global molecular feature  ====================
def calculate_global_features_mol(mol):
    """Calculate 25 optimized global molecular features using RDKit"""
    try:
        if mol is None or mol.GetNumAtoms() == 0:
            return [0.0] * 25

        mol_gb_feat = []
        mol_gb_feat.append(Descriptors.MolWt(mol))
        mol_gb_feat.append(Crippen.MolLogP(mol))
        mol_gb_feat.append(Crippen.MolMR(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcTPSA(mol))
        mol_gb_feat.append(Lipinski.NumHDonors(mol))
        mol_gb_feat.append(Lipinski.NumHAcceptors(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcHallKierAlpha(mol))
        mol_gb_feat.append(Descriptors.NumValenceElectrons(mol))

        radius_gyration = 0.0
        mol_h = Chem.AddHs(mol)
        if mol_h.GetNumConformers() == 0:
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = 0
                AllChem.EmbedMolecule(mol_h, params)
                AllChem.UFFOptimizeMolecule(mol_h)
            except:
                pass
        if mol_h.GetNumConformers() > 0:
            radius_gyration = rdMolDescriptors.CalcRadiusOfGyration(mol_h)

        mol_gb_feat.append(radius_gyration)
        mol_gb_feat.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcFractionCSP3(mol))
        mol_gb_feat.append(Lipinski.NumHeteroatoms(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcNumAmideBonds(mol))
        num_atoms = mol.GetNumAtoms()
        mol_gb_feat.append(num_atoms)
        mol_gb_feat.append(mol.GetNumBonds())
        mol_gb_feat.append(rdMolDescriptors.CalcNumRings(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        mol_gb_feat.append(Lipinski.NumHeteroatoms(mol) / max(1, num_atoms))
        mol_gb_feat.append(rdMolDescriptors.CalcNumRotatableBonds(mol) / max(1, num_atoms))
        mol_gb_feat.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
        mol_gb_feat.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))

        clean = []
        for x in mol_gb_feat:
            if x is None:
                clean.append(0.0)
                continue
            try:
                xf = float(x)
                clean.append(xf if np.isfinite(xf) else 0.0)
            except Exception:
                clean.append(0.0)
        mol_gb_feat = clean
        assert len(mol_gb_feat) == 25, f"Expected 25 features, got {len(mol_gb_feat)}"
        return mol_gb_feat
    except Exception as e:
        print(f"Global feature calculation failed: {e}")
        return [0.0] * 25

def get_global_feature_names():
    return [
        'Molecular_Weight', 'LogP', 'Molar_Refractivity',
        'TPSA', 'H_Bond_Donors', 'H_Bond_Acceptors',
        'HallKier_Alpha', 'Valence_Electrons',
        'Radius_of_Gyration', 'Rotatable_Bonds', 'Fraction_SP3',
        'Heteroatoms', 'Amide_Bonds',
        'Num_Atoms', 'Num_Bonds', 'Ring_Count',
        'Aromatic_Rings', 'Aliphatic_Rings',
        'Heteroatom_Ratio', 'Flexibility_Index',
        'Spiro_Atoms', 'Bridgehead_Atoms', 'Saturated_Rings',
        'Aliphatic_Carbocycles', 'Aliphatic_Heterocycles'
    ]

# ==================== smoles to graph  ====================
def smiles_to_graph(smiles, mol_scaler=None):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        AllChem.UFFOptimizeMolecule(mol)

        atom_features = get_atom_features(mol)
        x = torch.tensor(atom_features, dtype=torch.float)

        edge_index = []
        edge_attr = []
        bond_lengths = compute_bond_length(mol)
        for bond, bond_length in zip(mol.GetBonds(), bond_lengths):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index += [[i, j], [j, i]]
            bond_features = get_bond_features(bond, bond_length)
            edge_attr += [bond_features, bond_features]

        if len(edge_attr) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        mol_global_feat = np.array(calculate_global_features_mol(mol), dtype=np.float32)
        if mol_scaler is not None:
            mol_global_feat = mol_scaler.transform(mol_global_feat.reshape(1, -1)).ravel().astype(np.float32)
        mol_global_feat = torch.tensor(mol_global_feat, dtype=torch.float).unsqueeze(0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.mol_global_features = mol_global_feat
        return data

    except Exception as e:
        print(f"Error converting SMILES to graph: {e}")
        data = Data(
            x=torch.zeros((1, 22), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 6), dtype=torch.float)
        )
        data.mol_global_features = torch.zeros((1, 25), dtype=torch.float)
        return data

