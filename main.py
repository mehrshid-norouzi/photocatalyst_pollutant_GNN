"""
main.py

"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from rdkit import Chem
warnings.filterwarnings('ignore')
from torch_geometric.data import Batch, Data

from crystal_features import clean_photocat_name, cif_to_graph, extract_global_features, build_global_feature_vector
from molecule_features import smiles_to_graph, calculate_global_features_mol
from model import GNNModel
from visualization import (
    collect_predictions, plot_calculated_vs_experimental, compute_regression_stats,
    plot_combined_williams, plot_residuals, calculate_kunal_roy_validation,
    parity_plot_publication, plot_error_vs_k, plot_pca_combined,
    extract_all_features, plot_model_space_3d, plot_tsne_3d, plot_pca_umap_with_splits
)
import random, numpy as np, torch
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
from config import  CIF_DIR, EXCEL_PATH
from config import OUTPUT_DIR
# ----------------------------- scailer and other functions ----------------------------- #

def load_and_preprocess_data():
    """Load and preprocess the Excel data"""
    print("\n" + "="*60)
    print(" LOADING EXCEL FILE")
    print("="*60)
    excel_path = input('please input excel path:')
    if not excel_path:
        return None
    print(f"\n Loading file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
        print(" File loaded successfully!",
              "\n  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\n First 5 rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f" Error loading file: {e}")
        return None

def detect_columns(df):
    """Detect columns automatically based on common patterns"""
    print("\n detect columns automaticly...")
    columns = df.columns.tolist()
    # Find SMILES column
    smiles_candidates = []
    for col in columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['smiles', 'smile', 'smi', 'structure', 'mol']):
            smiles_candidates.append(col)
    if smiles_candidates:
        smiles_col = smiles_candidates[0]
        print(f"  SMILES column detected: {smiles_col}")
    else:
        print("\nAvailable columns:")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")
        while True:
            try:
                choice = int(input("\nEnter the number of the SMILES column: "))
                if 1 <= choice <= len(columns):
                    smiles_col = columns[choice-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(columns)}")
            except:
                print("Please enter a valid number")
    # Find Crystal column
    crystal_candidates = []
    for col in columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['Photocat', 'Photocatalyst', 'Photo', 'crystal']):
            crystal_candidates.append(col)
    if crystal_candidates:
        crystal_col = crystal_candidates[0]
        print(f"  Crystal column detected: {crystal_col}")
    else:
        print("\nAvailable columns:")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")
        while True:
            try:
                choice = int(input("\nEnter the number of the Crystal column: "))
                if 1 <= choice <= len(columns):
                    crystal_col = columns[choice-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(columns)}")
            except:
                print("Please enter a valid number")
    # Find target column (k)
    target_candidates = []
    for col in columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['k', 'target', 'rate', 'constant', 'degradation', 'reaction']):
            target_candidates.append(col)
    if target_candidates:
        target_col = target_candidates[0]
        print(f"  Target column detected: {target_col}")
    else:
        print("\nNumeric columns for target:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for i, col in enumerate(numeric_cols, 1):
            print(f"  {i:2d}. {col} (range: {df[col].min():.4f} to {df[col].max():.4f})")
        while True:
            try:
                choice = int(input("\nEnter the number of the target column (k): "))
                if 1 <= choice <= len(numeric_cols):
                    target_col = numeric_cols[choice-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(numeric_cols)}")
            except:
                print("Please enter a valid number")
    # Find experimental parameter columns
    print("\n Detecting experimental parameter columns...")
    param_patterns = {
        'dosage': ['dosage', 'dose', 'catalyst', 'amount', 'loading'],
        'concentration': ['concentration', 'conc', 'c0', 'initial'],
        'ph': ['ph', 'pH', 'acidity', 'alkalinity'],
        'light': ['light', 'illumination', 'intensity', 'irradiation', 'lamp']
    }
    detected_params = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for param_name, keywords in param_patterns.items():
        candidates = []
        for col in numeric_cols:
            if col == target_col or col == smiles_col or col == crystal_col:
                continue
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in keywords):
                candidates.append(col)
        if candidates:
            detected_params[param_name] = candidates[0]
            print(f"  {param_name}: {candidates[0]}")
        else:
            detected_params[param_name] = None
    missing_params = [k for k, v in detected_params.items() if v is None]
    if missing_params:
        print("\n The following columns were not detected:")
        for param in missing_params:
            print(f"  - {param}")
        print("\nRemaining numeric columns:")
        remaining_numeric = [col for col in numeric_cols 
                           if col not in [target_col, smiles_col] + list(filter(None, detected_params.values()))]
        for i, col in enumerate(remaining_numeric, 1):
            print(f"  {i:2d}. {col}")
        for param in missing_params:
            while True:
                try:
                    choice = input(f"\nEnter the column number for '{param}' (or press Enter to skip): ").strip()
                    if choice == "":
                        print(f"Column '{param}' skipped")
                        break
                    choice = int(choice)
                    if 1 <= choice <= len(remaining_numeric):
                        detected_params[param] = remaining_numeric[choice-1]
                        print(f"  {param}: {remaining_numeric[choice-1]}")
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(remaining_numeric)}")
                except:
                    print("Please enter a valid number")
    dosage_col = detected_params.get('dosage')
    concentration_col = detected_params.get('concentration')
    ph_col = detected_params.get('ph')
    light_col = detected_params.get('light')
    experimental_cols = [col for col in [dosage_col, concentration_col, ph_col, light_col] if col is not None]
    print("\n Detected columns:",
          f"\n SMILES: {smiles_col}",
          f"\n Target: {target_col}")
    if dosage_col:
        print(f"  Dosage: {dosage_col}")
    if concentration_col:
        print(f"  Concentration: {concentration_col}")
    if ph_col:
        print(f"  pH: {ph_col}")
    if light_col:
        print(f"  Light: {light_col}")
    return smiles_col, crystal_col, target_col, experimental_cols

def experimental_feats_processing(df, experimental_cols, exp_scaler=None, fit=False):
    """Process experimental features with train-only scaling."""
    feats_wo_light = df.loc[:, experimental_cols[:-1]].values.astype(np.float32)
    if exp_scaler is None:
        exp_scaler = StandardScaler()
    if fit:
        feats_wo_light = exp_scaler.fit_transform(feats_wo_light)
    else:
        feats_wo_light = exp_scaler.transform(feats_wo_light)
    light_raw = df.loc[:, experimental_cols[-1]].values
    light_list = []
    for v in light_raw:
        if v == 1:
            encoded_vec = [1, 0, 0]
        elif v == 2:
            encoded_vec = [0, 1, 0]
        elif v == 3:
            encoded_vec = [0, 0, 1]
        else:
            encoded_vec = [0, 0, 0]
            print("The light value was not defined!")
        light_list.append(encoded_vec)
    light_array = np.array(light_list, dtype=np.float32)
    exp_features = np.hstack([feats_wo_light, light_array]).astype(np.float32)
    return exp_features, exp_scaler

def fit_train_only_scalers(train_df, experimental_cols, smiles_col="smiles", crystal_col="Photocat", cif_dir=None):
    """Fit scalers ONLY on the training split."""
    exp_scaler = StandardScaler()
    exp_cont = train_df.loc[:, experimental_cols[:-1]].values.astype(np.float32)
    exp_scaler.fit(exp_cont)
    mol_scaler = StandardScaler()
    mol_rows = []
    for s in train_df[smiles_col].values:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError("Invalid SMILES")
            feats = np.array(calculate_global_features_mol(mol), dtype=np.float32)
        except Exception:
            feats = np.zeros((25,), dtype=np.float32)
        mol_rows.append(feats)
    mol_rows = np.vstack(mol_rows) if len(mol_rows) else np.zeros((1, 25), dtype=np.float32)
    mol_scaler.fit(mol_rows)
    crys_scaler = StandardScaler()
    crys_rows = []
    for name in train_df[crystal_col].apply(clean_photocat_name).values:
        try:
            if cif_dir is None:
                raise FileNotFoundError("cif_dir is None")
            cif_path = os.path.join(cif_dir, f"{name}.cif")
            if not os.path.exists(cif_path):
                raise FileNotFoundError(f"Missing CIF: {cif_path}")
            gdict = extract_global_features(cif_path, crystal_name=name)
            gf = build_global_feature_vector(gdict).astype(np.float32)
            feats18 = gf[:18]
        except Exception:
            feats18 = np.zeros((18,), dtype=np.float32)
        crys_rows.append(feats18)
    crys_rows = np.vstack(crys_rows) if len(crys_rows) else np.zeros((1, 18), dtype=np.float32)
    crys_scaler.fit(crys_rows)
    return exp_scaler, mol_scaler, crys_scaler

class GNNDataset(Dataset):
    def __init__(self, dataframe, numerical_features, smiles_col='smiles', crystal_col='Photocat', target_col='target', cif_dir=None,
                 exp_scaler=None, mol_scaler=None, crys_scaler=None, target_scaler=None):
        
        self.original_indices = dataframe.index.values   
        self.dataframe = dataframe.reset_index(drop=True)
        self.smiles = self.dataframe[smiles_col]
        self.crystals = self.dataframe[crystal_col].apply(clean_photocat_name)
        self.targets = self.dataframe[target_col].values.astype(np.float32)
        self.numerical_features = numerical_features
        self.cif_dir = cif_dir
        self.exp_scaler = exp_scaler
        self.mol_scaler = mol_scaler
        self.crys_scaler = crys_scaler
        exp_feats, _ = experimental_feats_processing(self.dataframe, self.numerical_features, exp_scaler=exp_scaler, fit=False)
        self.experimental_feats = exp_feats
        self.experimental_feats = torch.tensor(self.experimental_feats, dtype=torch.float)
        y = self.dataframe[target_col].values.reshape(-1, 1).astype(np.float32)
        if target_scaler is not None:
            y = target_scaler.transform(y)
        self.targets = torch.tensor(y, dtype=torch.float)
        self.target_scaler = target_scaler

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        smiles = self.smiles.iloc[idx]
        mol_graph = smiles_to_graph(smiles, mol_scaler=self.mol_scaler)
        if mol_graph is None:
            mol_graph = Data(x=torch.zeros((1, 22)), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 6)))
        crys_name = self.crystals.iloc[idx]
        crys_graph = cif_to_graph(crys_name, self.cif_dir, crys_scaler=self.crys_scaler)
        experimental_feat = self.experimental_feats[idx]
        target = self.targets[idx]
        original_idx = self.original_indices[idx]   
        return mol_graph, crys_graph, experimental_feat, target, original_idx

def collate_fn(batch):
    mol_graphs, crystal_graphs, experimental_feats, targets, indices = zip(*batch)
    batch_mol_graph = Batch.from_data_list(list(mol_graphs))
    batch_crystal_graph = Batch.from_data_list(list(crystal_graphs))
    experimental_feats = torch.stack(experimental_feats)
    targets = torch.stack(targets)
    indices = torch.tensor(indices, dtype=torch.long)
    return batch_mol_graph, batch_crystal_graph, experimental_feats, targets, indices
# ----------------------------- Run  ----------------------------- #
if __name__ == "__main__":
   
    cif_dir = CIF_DIR
    excel_path =EXCEL_PATH
    
    
    print(f"\n Loading file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
        print(" File loaded successfully!")
        print("\n First 5 rows:")
        print(df.head())
    except Exception as e:
        print(e)
        exit()
    
  
    smiles_col, crystal_col, target_col, experimental_cols = detect_columns(df)
    
   
    df_clean = pd.DataFrame({
        'smiles': df[smiles_col],
        'k': pd.to_numeric(df[target_col], errors='coerce'),
        'Photocat': df[crystal_col]
    })
    for i, col in enumerate(experimental_cols):
        df_clean[f'exp_{i}'] = pd.to_numeric(df[col], errors='coerce')
    initial_count = len(df_clean)
    df_clean = df_clean.dropna()
    final_count = len(df_clean)
    print(f"  Initial samples: {initial_count}")
    print(f"  After cleaning: {final_count}")
    print(f"  Removed: {initial_count - final_count} rows with NaN values")
    
    
    numerical_features = [f'exp_{i}' for i in range(len(experimental_cols))]
    
    # ==================== split data ====================
    forced_train_indices = [67, 146, 71, 282, 114, 42, 127, 343, 348, 173, 94]  # 0-based indices
   
    missing = [idx for idx in forced_train_indices if idx not in df_clean.index]
    if missing:
        raise ValueError(f"no index finding: {missing}")
    forced_train_df = df_clean.loc[forced_train_indices]
    remaining_df = df_clean.drop(forced_train_indices)
    N = len(df_clean)
    k = len(forced_train_indices)
    target_train_size = int(round(0.7 * N))
    target_val_size   = int(round(0.15 * N))
    target_test_size  = N - target_train_size - target_val_size
    if k >= target_train_size:
        print(f"The number of required indices ({k}) is greater than the target Train size ({target_train_size}).")
        train_df = forced_train_df
        val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)
    else:
        needed_from_rest = target_train_size - k
        rest_size = len(remaining_df)
        proportion_val_test = (target_val_size + target_test_size) / rest_size
        train_rest_df, temp_df = train_test_split(remaining_df, test_size=proportion_val_test, random_state=42, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        train_df = pd.concat([forced_train_df, train_rest_df])
    train_idx = [i + 1 for i in train_df.index.tolist()]
    val_idx   = [i + 1 for i in val_df.index.tolist()]
    test_idx  = [i + 1 for i in test_df.index.tolist()]
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    forced_in_train = all(idx in train_df.index for idx in forced_train_indices)
    print(f"All forced indices are in train: {forced_in_train}")
    
    # ==================== fit train scailer ====================
    print("\n Creating datasets...")
    exp_scaler, mol_scaler, crys_scaler = fit_train_only_scalers(
        train_df=train_df,
        experimental_cols=numerical_features,
        smiles_col=smiles_col,
        crystal_col=crystal_col,
        cif_dir=cif_dir,
    )
    target_scaler = StandardScaler()
    target_scaler.fit(train_df[target_col].values.reshape(-1, 1))
    
    # ==================== datat set and loader ====================
    train_dataset = GNNDataset(train_df, numerical_features, smiles_col,
                                crystal_col, target_col, cif_dir=cif_dir,
                                exp_scaler=exp_scaler, mol_scaler=mol_scaler, crys_scaler=crys_scaler, target_scaler=target_scaler)
    val_dataset = GNNDataset(val_df, numerical_features, smiles_col,
                                crystal_col, target_col, cif_dir=cif_dir,
                                exp_scaler=exp_scaler, mol_scaler=mol_scaler, crys_scaler=crys_scaler, target_scaler=target_scaler)
    test_dataset = GNNDataset(test_df, numerical_features, smiles_col,
                                crystal_col, target_col, cif_dir=cif_dir,
                                exp_scaler=exp_scaler, mol_scaler=mol_scaler, crys_scaler=crys_scaler, target_scaler=target_scaler)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(42), collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # ==================== model craeting====================
    print("\n Initializing model...")
    experimental_input_dim = train_dataset[0][2].shape[0]
    global_feat_dim = train_dataset[0][1].global_features.shape[1]
    model = GNNModel(
        mol_input_dim=22,
        crystal_atom_dim=92,
        experimental_input_dim=experimental_input_dim,
        global_feat_dim=global_feat_dim,
        mol_global_feat_dim=25,
        mol_hidden_dim=64,
        crystal_hidden_dim=64,
        experimental_hidden_dim=64,
        combined_hidden_dim=256,
        fusion_dim=64,
        feature_mask={
            "mol_graph": True,
            "mol_global": True,
            "crystal_graph": True,
            "crystal_global": True,
            "experimental": True
        },
        interaction_mask={
            "mol_exp": False,
            "crys_exp": False,
            "mol_crys": False,
            "triple": False
        }
    )
    mask_status = {
        "molecular_graph": model.feature_mask["mol_graph"],
        "molecular_global": model.feature_mask["mol_global"],
        "crystal_graph": model.feature_mask["crystal_graph"],
        "crystal_global": model.feature_mask["crystal_global"],
        "experimental": model.feature_mask["experimental"]
    }
    print("Active branches:", [k for k, v in mask_status.items() if v])
    print("Inactive branches:", [k for k, v in mask_status.items() if not v])
    interaction_status = {
        "mol_exp": model.interaction_mask["mol_exp"],
        "crys_exp": model.interaction_mask["crys_exp"],
        "mol_crys": model.interaction_mask["mol_crys"],
        "triple": model.interaction_mask["triple"]
    }
    print("Active interactions:", [k for k, v in interaction_status.items() if v])
    print("Inactive interactions:", [k for k, v in interaction_status.items() if not v])
    print("Fusion input dimension:", model.fc1.in_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Model initialized on {device}")
    print(f"  Using crystal data: {cif_dir is not None}")
    print("  Using band gap data: ✓")
    
    # ==================== training model ====================
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    num_epochs = 100
    PATIENCE = 50
    best_val_loss = float('inf')
    patience_counter = 0
    # #Training loop
    # train_losses_history = []
    # val_losses_history = []
    # print(f"\n Starting training for {num_epochs} epochs...")
    # print(f"  Device: {device}")
    # print(f"  Batch size: {batch_size}")
    # print(f"  Early stopping patience: {PATIENCE}")
    # print("  Features included: Molecular, Crystal, Band Gap, Experimental")
    # print("-" * 50)
    # for epoch in range(1, num_epochs + 1):
    #     model.train()
    #     train_losses = []
    #     for mol_graphs, crystal_graphs, exp_feats, targets , _ in train_loader:
    #         mol_graphs = mol_graphs.to(device)
    #         exp_feats = exp_feats.to(device)
    #         targets = targets.to(device)
    #         if crystal_graphs is not None:
    #             crystal_graphs = crystal_graphs.to(device)
    #         optimizer.zero_grad()
    #         outputs, _ = model(mol_graphs, crystal_graphs, exp_feats)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()
    #         train_losses.append(loss.item())
    #     avg_train_loss = np.mean(train_losses) if train_losses else 0
    #     train_losses_history.append(avg_train_loss)
    #     model.eval()
    #     val_losses = []
    #     with torch.no_grad():
    #         for mol_graphs, crystal_graphs, exp_feats, targets , _  in val_loader:
    #             mol_graphs = mol_graphs.to(device)
    #             exp_feats = exp_feats.to(device)
    #             targets = targets.to(device)
    #             if crystal_graphs is not None:
    #                 crystal_graphs = crystal_graphs.to(device)
    #             outputs, _ = model(mol_graphs, crystal_graphs, exp_feats)
    #             loss = criterion(outputs, targets)
    #             val_losses.append(loss.item())
    #     avg_val_loss = np.mean(val_losses) if val_losses else 0
    #     val_losses_history.append(avg_val_loss)
    #     scheduler.step(avg_val_loss)
    #     if epoch % 10 == 0 or epoch == 1:
    #         current_lr = optimizer.param_groups[0]['lr']
    #         print(f"Epoch {epoch:03d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pth')
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= PATIENCE:
    #             print(f" Early stopping triggered at epoch {epoch}")
    #             break
    
  
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_losses_history, label='Train Loss', linewidth=2)
    # plt.plot(val_losses_history, label='Validation Loss', linewidth=2)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Loss', fontsize=14)
    # plt.title('Training History', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # train_rmse = np.sqrt(train_losses_history)
    # val_rmse = np.sqrt(val_losses_history)
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_rmse, label='Train RMSE', linewidth=2)
    # plt.plot(val_rmse, label='Validation RMSE', linewidth=2)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('RMSE', fontsize=14)
    # plt.title('Training History — RMSE', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'training_history_rmse.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    #load best model 
    print("\n Loading best model...")
    
    best_model_path = OUTPUT_DIR / 'best_model.pth'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        print("  Best model loaded successfully")
    else:
        print("  Warning: Best model not found, using current model")
    # ==================== Collecting predictions ====================
    print("\n Collecting predictions...")
   
    train_pred_scaled, train_true_scaled, train_pred_real, train_true_real, \
    train_exp_feats, train_combined_feats, train_loss, train_r2, train_indices = collect_predictions(
        train_loader, model, device, criterion, target_scaler
    )
    
    val_pred_scaled, val_true_scaled, val_pred_real, val_true_real, \
    val_exp_feats, val_combined_feats, val_loss, val_r2, val_indices = collect_predictions(
        val_loader, model, device, criterion, target_scaler
    )
    
    test_pred_scaled, test_true_scaled, test_pred_real, test_true_real, \
    test_exp_feats, test_combined_feats, test_loss, test_r2, test_indices = collect_predictions(
        test_loader, model, device, criterion, target_scaler
    )
    
    train_out = (train_pred_scaled, train_true_scaled, train_pred_real, train_true_real,
                 train_exp_feats, train_combined_feats, train_loss, train_r2)
    
    val_out = (val_pred_scaled, val_true_scaled, val_pred_real, val_true_real,
               val_exp_feats, val_combined_feats, val_loss, val_r2)
    
    test_out = (test_pred_scaled, test_true_scaled, test_pred_real, test_true_real,
                test_exp_feats, test_combined_feats, test_loss, test_r2)
    # ==================== plots ====================
    # Parity plot
    from visualization import parity_plot_from_collect
    parity_plot_from_collect(test_out, "Test Set Parity Plot")
    
    # Williams plot
    plot_combined_williams(
        train_combined_feats,          # train:  (combined features)
        val_combined_feats,            # val
        test_combined_feats,           # test
        train_pred_scaled,             # train:  (scaled)
        val_pred_scaled,               # val
        test_pred_scaled,              # test
        train_true_scaled,             # train:  (scaled)
        val_true_scaled,               # val
        test_true_scaled,              # test
        train_indices,                 # train: ( collect_predictions)
        val_indices,                   # val
        test_indices                   # test
    )
    # Residuals plots
    plot_residuals(test_true_scaled.flatten(), test_pred_scaled.flatten(), 'testset')
    plot_residuals(train_true_scaled.flatten(), train_pred_scaled.flatten(), 'trainset')
    plot_residuals(val_true_scaled.flatten(), val_pred_scaled.flatten(), 'valset')
    
    # Calculated vs Experimental plots
    results = []
    dsname = []
    datasets = [
        (train_pred_scaled, train_true_scaled, train_pred_real, train_true_real, "Train", train_idx),
        (val_pred_scaled, val_true_scaled, val_pred_real, val_true_real, "Validation", val_idx),
        (test_pred_scaled, test_true_scaled, test_pred_real, test_true_real, "Test", test_idx)
    ]
    for pred, tgt, pred_real, true_real, name, idx in datasets:
        if len(pred) > 0 and len(tgt) > 0:
            slope, intercept, slope_sd, intercept_sd, result = compute_regression_stats(tgt.flatten(), pred.flatten())
            results.append(result)
            dsname.append(name)
            plot_calculated_vs_experimental(pred_real.flatten(), true_real.flatten(), name, idx, 
                                             slope, intercept, slope_sd, intercept_sd)
    
    # Roy validation
    train_roy = calculate_kunal_roy_validation(train_true_real, train_pred_real)
    val_roy = calculate_kunal_roy_validation(val_true_real, val_pred_real)
    test_roy = calculate_kunal_roy_validation(test_true_real, test_pred_real)
    print("\n[TRAIN SET]")
    for metric, value in train_roy.items():
        print(f"  {metric:20}: {value}")
    print("\n[VALIDATION SET]")
    for metric, value in val_roy.items():
        print(f"  {metric:20}: {value}")
    print("\n[TEST SET]")
    for metric, value in test_roy.items():
        print(f"  {metric:20}: {value}")
    
    # Publication parity plot
    parity_plot_publication(test_true_real.flatten(), test_pred_real.flatten(), "Parity Plot (Test Set)", "Parity_Test.png")
    plot_error_vs_k(test_true_real.flatten(), test_pred_real.flatten())
    plot_pca_combined(train_combined_feats, test_combined_feats)
    
    # save regretion results
    if results:
        results_df = pd.DataFrame(results, index=dsname)
        results_df.to_excel('Regression_results.xlsx')
        print("  Regression results saved to 'Regression_results.xlsx'")
    
    # ====================  PCA/UMAP/t-SNE  ====================
    print("\n" + "="*70)
    print(" PCA & UMAP FOR MODEL'S LEARNED SPACE (combined)")
    print("="*70)
    features_dict_train, targets_train_real, indices_train, _ = extract_all_features(
        model, device, train_loader, target_scaler, df_clean, "Train")
    features_dict_val, targets_val_real, indices_val, _ = extract_all_features(
        model, device, val_loader, target_scaler, df_clean, "Validation")
    features_dict_test, targets_test_real, indices_test, _ = extract_all_features(
        model, device, test_loader, target_scaler, df_clean, "Test")
    
    
    
    # ==================== excel report====================
    print("\n Creating detailed Excel reports...")
    val_results_df = pd.DataFrame({
        "Excel_Row_Index": np.array(val_idx).flatten(),
        "True_k_log": val_true_real.flatten(),
        "Predicted_k_log": val_pred_real.flatten(),
        "Error": val_pred_real.flatten() - val_true_real.flatten(),
        "Absolute_Error": np.abs(val_pred_real.flatten() - val_true_real.flatten())
    }).sort_values(by="Absolute_Error", ascending=False)
    val_results_df.to_excel("Validation_Detailed_Results.xlsx", index=False)
    print(" Validation report saved to: Validation_Detailed_Results.xlsx")
    
    train_results_df = pd.DataFrame({
        "Excel_Row_Index": np.array(train_idx).flatten(),
        "True_k_log": train_true_real.flatten(),
        "Predicted_k_log": train_pred_real.flatten(),
        "Error": train_pred_real.flatten() - train_true_real.flatten(),
        "Absolute_Error": np.abs(train_pred_real.flatten() - train_true_real.flatten())
    }).sort_values(by="Absolute_Error", ascending=False)
    train_results_df.to_excel("train_Detailed_Results.xlsx", index=False)
    print(" Train report saved to: train_Detailed_Results.xlsx")
    
    test_results_df = pd.DataFrame({
        "Excel_Row_Index": np.array(test_idx).flatten(),
        "True_k_log": test_true_real.flatten(),
        "Predicted_k_log": test_pred_real.flatten(),
        "Error": test_pred_real.flatten() - test_true_real.flatten(),
        "Absolute_Error": np.abs(test_pred_real.flatten() - test_true_real.flatten())
    }).sort_values(by="Absolute_Error", ascending=False)
    test_results_df.to_excel("test_Detailed_Results.xlsx", index=False)
    print(" Test report saved to: test_Detailed_Results.xlsx")
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
# ==================== استخراج ویژگی‌ها برای PCA/t-SNE ====================
print("\n Extracting features for visualization...")

features_train, targets_train_real, indices_train, _ = extract_all_features(
    model=model, device=device, loader=train_loader,
    target_scaler=target_scaler, original_df=df_clean, split_name="Train"
)

features_val, targets_val_real, indices_val, _ = extract_all_features(
    model=model, device=device, loader=val_loader,
    target_scaler=target_scaler, original_df=df_clean, split_name="Validation"
)

features_test, targets_test_real, indices_test, _ = extract_all_features(
    model=model, device=device, loader=test_loader,
    target_scaler=target_scaler, original_df=df_clean, split_name="Test"
)

# ==================== PCA/UMAP/t-SNE on learned space ====================
plot_model_space_3d(
    features_dict_train=features_train,
    features_dict_val=features_val,
    features_dict_test=features_test,
    targets_train=targets_train_real,
    targets_val=targets_val_real,
    targets_test=targets_test_real,
    indices_train=indices_train,
    indices_val=indices_val,
    indices_test=indices_test,
    splits_train=np.array(['Train'] * len(features_train['combined'])),
    splits_val=np.array(['Validation'] * len(features_val['combined'])),
    splits_test=np.array(['Test'] * len(features_test['combined'])),
    original_df=df_clean,
    feature_type="combined",
    numerical_features=numerical_features
)

plot_tsne_3d(
    features_dict_train=features_train,
    features_dict_val=features_val,
    features_dict_test=features_test,
    targets_train=targets_train_real,
    targets_val=targets_val_real,
    targets_test=targets_test_real,
    indices_train=indices_train,
    indices_val=indices_val,
    indices_test=indices_test,
    splits_train=np.array(['Train'] * len(features_train['combined'])),
    splits_val=np.array(['Validation'] * len(features_val['combined'])),
    splits_test=np.array(['Test'] * len(features_test['combined'])),
    original_df=df_clean,
    feature_type="combined",
    numerical_features=numerical_features
)

plot_pca_umap_with_splits(
    feature_type="combined",

    features_dict_train=features_train,
    features_dict_val=features_val,
    features_dict_test=features_test,

    targets_train=targets_train_real,
    targets_val=targets_val_real,
    targets_test=targets_test_real,

    indices_train=indices_train,
    indices_val=indices_val,
    indices_test=indices_test,

    original_df=df_clean,

    experimental_cols=numerical_features
)
#==========================================================================
print("\n Creating detailed Validation Excel report...")


val_pred_flat = np.array(val_pred_real).flatten()
val_true_flat = np.array(val_true_real).flatten()
val_idx_flat = np.array(val_idx).flatten()


error = val_pred_flat - val_true_flat
abs_error = np.abs(error)


val_results_df = pd.DataFrame({
    "Excel_Row_Index": val_idx_flat,
    "True_k_log": val_true_flat,
    "Predicted_k_log": val_pred_flat,
    "Error": error,
    "Absolute_Error": abs_error
})


val_results_df = val_results_df.sort_values(
    by="Absolute_Error",
    ascending=False
)


output_file = "Validation_Detailed_Results.xlsx"
val_results_df.to_excel(output_file, index=False)

print(f" Validation report saved to: {output_file}")

print("\n Creating detailed train Excel report...")


train_pred_flat = np.array(train_pred_real).flatten()
train_true_flat = np.array(train_true_real).flatten()
train_idx_flat = np.array(train_idx).flatten()


error = train_pred_flat - train_true_flat
abs_error = np.abs(error)


train_results_df = pd.DataFrame({
    "Excel_Row_Index": train_idx_flat,
    "True_k_log": train_true_flat,
    "Predicted_k_log": train_pred_flat,
    "Error": error,
    "Absolute_Error": abs_error
})


train_results_df = train_results_df.sort_values(
    by="Absolute_Error",
    ascending=False
)


output_file = "train_Detailed_Results.xlsx"
train_results_df.to_excel(output_file, index=False)

print(f" train report saved to: {output_file}")


print("\n Creating detailed test Excel report...")


test_pred_flat = np.array(test_pred_real).flatten()
test_true_flat = np.array(test_true_real).flatten()
test_idx_flat = np.array(test_idx).flatten()


error = test_pred_flat - test_true_flat
abs_error = np.abs(error)


test_results_df = pd.DataFrame({
    "Excel_Row_Index": test_idx_flat,
    "True_k_log": test_true_flat,
    "Predicted_k_log": test_pred_flat,
    "Error": error,
    "Absolute_Error": abs_error
})


test_results_df = test_results_df.sort_values(
    by="Absolute_Error",
    ascending=False
)


output_file = "test_Detailed_Results.xlsx"
test_results_df.to_excel(output_file, index=False)

print(f" test report saved to: {output_file}")


print("\n" + "=" * 70)
print("              AI PHOTOCATALYST MODELING FRAMEWORK")
print("=" * 70)
print("Research Team:")
print("  • Mehrshid Norouzi Zarmehri")
print("  • Mahya Vazifeh Solout")
print()
print("Academic Supervisor:")
print("  • Prof.Jahan B. Ghasemi")

  
print("=" * 70)
print("                  END OF PROGRAM")
print("=" * 70)
from config import move_outputs_to_output_folder

move_outputs_to_output_folder()