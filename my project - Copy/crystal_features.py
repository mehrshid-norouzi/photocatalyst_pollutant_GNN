"""
crystal_features.py

"""

import os
import json
import numpy as np
from scipy.spatial import KDTree
from typing import List, Dict, Any
import torch
from torch_geometric.data import Data
import pandas as pd
import random, numpy as np, torch
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from deepchem.utils.data_utils import get_data_dir
from deepchem.feat.material_featurizers.cgcnn_featurizer import CGCNNFeaturizer
#=====================================================================
from config import DATA_DIR, CIF_DIR, ATOM_INIT_PATH
# ==================== constanats ====================
COVALENT_RADII = {
    'H': 0.31,   'He': 0.28,
    'Li': 1.28,  'Be': 0.96,  'B': 0.84,   'C': 0.76,   'N': 0.71,   'O': 0.66,   'F': 0.57,   'Ne': 0.58,
    'Na': 1.66,  'Mg': 1.41,  'Al': 1.21,  'Si': 1.11,  'P': 1.07,   'S': 1.05,   'Cl': 1.02,  'Ar': 1.06,
    'K': 2.03,   'Ca': 1.76,  'Sc': 1.70,  'Ti': 1.60,  'V': 1.53,   'Cr': 1.39,  'Mn': 1.39,  'Fe': 1.32,
    'Co': 1.26,  'Ni': 1.24,  'Cu': 1.32,  'Zn': 1.22,  'Ga': 1.22,  'Ge': 1.20,  'As': 1.19,  'Se': 1.20,
    'Br': 1.20,  'Kr': 1.16,
    'Rb': 2.20,  'Sr': 1.95,  'Y': 1.90,   'Zr': 1.75,  'Nb': 1.64,  'Mo': 1.54,  'Tc': 1.47,  'Ru': 1.46,
    'Rh': 1.42,  'Pd': 1.39,  'Ag': 1.45,  'Cd': 1.44,  'In': 1.42,  'Sn': 1.39,  'Sb': 1.39,  'Te': 1.38,
    'I': 1.39,   'Xe': 1.40,
    'Cs': 2.44,  'Ba': 2.15,  'La': 2.07,  'Ce': 2.04,  'Pr': 2.03,  'Nd': 2.01,  'Pm': 1.99,  'Sm': 1.98,
    'Eu': 1.98,  'Gd': 1.96,  'Tb': 1.94,  'Dy': 1.92,  'Ho': 1.92,  'Er': 1.89,  'Tm': 1.90,  'Yb': 1.87,
    'Lu': 1.87,
    'Hf': 1.75,  'Ta': 1.70,  'W': 1.62,   'Re': 1.51,  'Os': 1.44,  'Ir': 1.41,  'Pt': 1.36,  'Au': 1.36,
    'Hg': 1.32,
    'Tl': 1.45,  'Pb': 1.46,  'Bi': 1.48,  'Po': 1.40,  'At': 1.50,  'Rn': 1.50,
    'Fr': 2.60,  'Ra': 2.21,  'Ac': 2.15,  'Th': 2.06,  'Pa': 2.00,  'U': 1.96,   'Np': 1.90,  'Pu': 1.87,
    'Am': 1.80,  'Cm': 1.69,  'Bk': 1.60,  'Cf': 1.60,  'Es': 1.60,  'Fm': 1.60,  'Md': 1.60,  'No': 1.60,
    'Lr': 1.60,
    'Rf': 1.60,  'Db': 1.60,  'Sg': 1.60,  'Bh': 1.60,  'Hs': 1.60,  'Mt': 1.60,  'Ds': 1.60,  'Rg': 1.60,
    'Cn': 1.60,  'Nh': 1.60,  'Fl': 1.60,  'Mc': 1.60,  'Lv': 1.60,  'Ts': 1.60,  'Og': 1.60,
}

BAND_GAP_DATABASE = {
    'ZnO': 0.72,
    'SnO2': 0.65,
    'Fe2O3': 0.82,
    'TiO2': 2.06,
    'WO3': 1.42,
    'MnO2': 1.15
}

CRYSTAL_SYSTEMS = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
POINT_GROUPS = [
    "1", "-1", "2", "m", "2/m", 
    "222", "mm2", "mmm", 
    "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm", 
    "3", "-3", "32", "3m", "-3m", 
    "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
    "23", "m-3", "432", "-43m", "m-3m"
]

# ==================== functions ====================
def clean_photocat_name(name):
    """Clean photocatalyst names to match CIF file names"""
    if pd.isna(name):
        return ""
    cleaned = str(name).replace('\n', '').replace('<br>', '').strip()
    replacements = {
        'Fe O2 3': 'Fe2O3',
        'Fe O 2 3': 'Fe2O3',
        'SnO2': 'SnO2',
        'SnO 2': 'SnO2',
        'TiO2': 'TiO2',
        'TiO 2': 'TiO2',
        'WO3': 'WO3',
        'WO 3': 'WO3',
        'MnO2': 'MnO2',
        'MnO 2': 'MnO2',
        'ZnO': 'ZnO'
    }
    for key, value in replacements.items():
        if key in cleaned:
            return value
    cleaned = cleaned.replace(' ', '')
    return cleaned

def atomic_volume(element_symbol: str) -> float:
    r = COVALENT_RADII.get(element_symbol, 1.0)
    return (4/3) * np.pi * (r ** 3)

def compute_packing_density(structure: Structure) -> float:
    total_volume = 0.0
    for site in structure.sites:
        elem = site.specie.symbol
        total_volume += atomic_volume(elem)
    return total_volume / structure.lattice.volume

def get_band_gap(crystal_name: str) -> float:
    cleaned_name = clean_photocat_name(crystal_name)
    return BAND_GAP_DATABASE.get(cleaned_name, 0.0)

def compute_avg_bond_length(structure: Structure, cutoff: float = 3.0) -> float:
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    bond_lengths = []
    for i, pos in enumerate(positions):
        neighbors = [j for j in kdtree.query_ball_point(pos, cutoff) if j != i]
        for j in neighbors:
            bond_lengths.append(np.linalg.norm(pos - positions[j]))
    return np.mean(bond_lengths) if bond_lengths else 0.0

def compute_avg_coordination_number(structure: Structure, cutoff: float = 3.0) -> float:
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    coordination_numbers = []
    for i, pos in enumerate(positions):
        indices = kdtree.query_ball_point(pos, cutoff)
        coordination_numbers.append(len(indices) - 1)
    return np.mean(coordination_numbers) if coordination_numbers else 0.0

def compute_avg_bond_angle(structure: Structure, cutoff: float = 3.0) -> float:
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    angles = []
    for i, pos in enumerate(positions):
        neighbors = [positions[j] for j in kdtree.query_ball_point(pos, cutoff) if j != i]
        for k in range(len(neighbors)):
            for l in range(k + 1, len(neighbors)):
                vec1 = neighbors[k] - pos
                vec2 = neighbors[l] - pos
                cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cosine = np.clip(cosine, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cosine)))
    return np.mean(angles) if angles else 0.0

def compute_polyhedra_shape_index(structure: Structure, cutoff: float = 3.0) -> float:
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    shape_indices = []
    for i, pos in enumerate(positions):
        neighbors = [positions[j] for j in kdtree.query_ball_point(pos, cutoff) if j != i]
        angles = []
        for k in range(len(neighbors)):
            for l in range(k + 1, len(neighbors)):
                vec1 = neighbors[k] - pos
                vec2 = neighbors[l] - pos
                cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cosine = np.clip(cosine, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cosine)))
        if angles:
            shape_indices.append(np.std(angles))
    return np.mean(shape_indices) if shape_indices else 0.0

def compute_avg_en_diff(structure: Structure, cutoff: float = 3.0) -> float:
    positions = np.array([site.coords for site in structure.sites])
    kdtree = KDTree(positions)
    en_diffs = []
    for i, site in enumerate(structure.sites):
        en_center = Element(site.specie.symbol).X or 0.0
        neighbors = [j for j in kdtree.query_ball_point(site.coords, cutoff) if j != i]
        if neighbors:
            en_neighbors = [Element(structure.sites[j].specie.symbol).X or 0.0 for j in neighbors]
            en_diffs.append(abs(en_center - np.mean(en_neighbors)))
    return np.mean(en_diffs) if en_diffs else 0.0

def compute_num_distinct_wyckoffs(structure: Structure) -> float:
    try:
        sga = SpacegroupAnalyzer(structure)
        sym = sga.get_symmetry_dataset()
        return float(len(set(sym.get('wyckoffs', [])))) if sym else 0.0
    except:
        return 0.0

def extract_global_features(cif_file: str, crystal_name: str = "") -> Dict[str, Any]:
    try:
        structure = Structure.from_file(cif_file)
    except Exception as e:
        print(f"Could not read CIF file {cif_file}: {e}")
        return {}
    
    lat = structure.lattice
    a, b, c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma
    volume = lat.volume
    packing_density = compute_packing_density(structure)
    ratio_c_a = c / a
    ratio_b_a = b / a
    num_atoms = structure.num_sites

    avg_bond_length = compute_avg_bond_length(structure)
    avg_coordination_number = compute_avg_coordination_number(structure)
    avg_bond_angle = compute_avg_bond_angle(structure)
    polyhedra_shape_index = compute_polyhedra_shape_index(structure)
    num_distinct_wyckoffs = compute_num_distinct_wyckoffs(structure)
    try:
        sga = SpacegroupAnalyzer(structure)
        pg = sga.get_point_group_symbol() or ''
        cs = sga.get_crystal_system() or ''
    except:
        pg, cs = '', ''
    
    avg_en_diff = compute_avg_en_diff(structure)
    band_gap = get_band_gap(crystal_name) if crystal_name else 0.0

    return {
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'volume': volume,
        'packing_density': packing_density,
        'ratio_c_a': ratio_c_a, 'ratio_b_a': ratio_b_a,
        'num_atoms': num_atoms,
        'avg_bond_length': avg_bond_length,
        'avg_coordination_number': avg_coordination_number,
        'avg_bond_angle': avg_bond_angle,
        'num_distinct_wyckoffs': num_distinct_wyckoffs,
        'avg_en_diff': avg_en_diff,
        'polyhedra_shape_index': polyhedra_shape_index,
        'band_gap': band_gap,
        'point_group': pg,
        'crystal_system': cs
    }

def one_hot_encode(value: str, categories: List[str]) -> np.ndarray:
    vec = np.zeros(len(categories))
    try:
        vec[categories.index(value.lower())] = 1.0
    except ValueError:
        pass
    return vec

def build_global_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    continuous = np.array([
        features['a'], features['b'], features['c'],
        features['alpha'], features['beta'], features['gamma'],
        features['volume'], features['packing_density'],
        features['ratio_c_a'], features['ratio_b_a'],
        features['num_atoms'],
        features['avg_bond_length'], features['avg_coordination_number'],
        features['avg_bond_angle'],
        features['num_distinct_wyckoffs'],
        features['avg_en_diff'], features['polyhedra_shape_index'],
        features['band_gap']
    ], dtype=np.float32)

    cs_vec = one_hot_encode(features['crystal_system'], CRYSTAL_SYSTEMS)
    pg_vec = one_hot_encode(features['point_group'], POINT_GROUPS)
    return np.concatenate([continuous, cs_vec, pg_vec])

# ==================== CGCNN  ====================

class OfflineCGCNNFeaturizer(CGCNNFeaturizer):
    def __init__(self, radius: float = 8.0, max_neighbors: int = 12, step: float = 0.2):
        self.radius = radius
        self.max_neighbors = int(max_neighbors)
        self.step = step
        # data_dir = get_data_dir()
        
        data_dir = DATA_DIR
        json_path = ATOM_INIT_PATH
        
        json_path = os.path.join(data_dir, "atom_init.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Need atom_init.json in {data_dir}. "
                "Please download and place it there."
            )
        with open(json_path, "r") as f:
            atom_init = json.load(f)
        self.atom_features = {
            int(k): np.array(v, dtype=np.float32) for k, v in atom_init.items()
        }
        self.valid_atom_number = set(self.atom_features.keys())

# ====================  CIF to graph ====================
def cif_to_graph(crys_name, data_dir, crys_scaler=None):
    featurizer = OfflineCGCNNFeaturizer()
    cif = os.path.join(data_dir, f"{crys_name}.cif")
    
    if not os.path.exists(cif):
        print(f"Warning: CIF file not found: {cif}")
        data = Data(
            x=torch.zeros((1, 92), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 41), dtype=torch.float)
        )
        gf = build_global_feature_vector(extract_global_features(cif))
        data.global_features = torch.tensor(gf, dtype=torch.float).unsqueeze(0)
        return data
    
    try:
        raw = featurizer.featurize([Structure.from_file(cif)])
        feat = raw
        while isinstance(feat, (np.ndarray, list)):
            feat = feat[0] if len(feat) > 0 else None
        if feat is None:
            data = Data(
                x=torch.zeros((1, 92), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 41), dtype=torch.float)
            )
        else:
            data = feat.to_pyg_graph()
        
        global_features_dict = extract_global_features(cif, crystal_name=crys_name)
        gf = build_global_feature_vector(global_features_dict)
        if crys_scaler is not None:
            gf_first18 = gf[:18].reshape(1, -1).astype(np.float32)
            gf[:18] = crys_scaler.transform(gf_first18).ravel()
        data.global_features = torch.tensor(gf, dtype=torch.float).unsqueeze(0)
        return data
    except Exception as e:
        print(f"Error processing CIF file {cif}: {e}")
        data = Data(
            x=torch.zeros((1, 92), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 41), dtype=torch.float)
        )
        data.global_features = torch.tensor(gf, dtype=torch.float).unsqueeze(0)
        return data