"""
visualization.py
 PCA/UMAP/t-SNE 
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import norm
from umap import UMAP
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ----------------------------- collect_predictions ----------------------------- #
def collect_predictions(loader, model, device, criterion, target_scaler=None):
    losses = []
    predictions = []
    targets_list = []
    experimental_feats = []
    combined_feats = []
    indices_list = []  

    model.eval()
    with torch.no_grad():
        for mol_graphs, crystal_graphs, exp_feats, tgt, idxs in loader:
            mol_graphs = mol_graphs.to(device)
            exp_feats = exp_feats.to(device)
            tgt = tgt.to(device)
            if crystal_graphs is not None:
                crystal_graphs = crystal_graphs.to(device)

            outputs, combined = model(mol_graphs, crystal_graphs, exp_feats)

            loss = criterion(outputs, tgt)
            losses.append(loss.item())

            predictions.append(outputs.cpu().numpy())
            targets_list.append(tgt.cpu().numpy())
            experimental_feats.append(exp_feats.cpu().numpy())
            combined_feats.append(combined.cpu().numpy())
            indices_list.append(idxs.cpu().numpy())   

    predictions = np.vstack(predictions)
    targets = np.vstack(targets_list)
    experimental_feats = np.vstack(experimental_feats)
    combined_feats = np.vstack(combined_feats)
    indices = np.concatenate(indices_list)          

    avg_loss = np.mean(losses)

    if target_scaler is not None:
        y_pred_real = target_scaler.inverse_transform(predictions)
        y_true_real = target_scaler.inverse_transform(targets)
        r2 = r2_score(y_true_real, y_pred_real)
    else:
        y_pred_real = predictions
        y_true_real = targets
        r2 = r2_score(targets, predictions)

    print(f" - Loss: {avg_loss:.4f} - R² (real scale): {r2:.4f}")

    return (
        predictions,        # scaled
        targets,            # scaled
        y_pred_real,        # real scale
        y_true_real,        # real scale
        experimental_feats,
        combined_feats,
        avg_loss,
        r2,
        indices             
    )
# ----------------------------- plots  ----------------------------- #
def plot_calculated_vs_experimental(predicted, actual, dataset_name, label, slope, intercept, slope_sd, intercept_sd):
    """
    Generates and displays a scatter plot comparing calculated and experimental k values
    """
    # Linear regression for confidence intervals
    
    reg = LinearRegression().fit(actual.reshape(-1, 1), predicted)
    y_fit = reg.predict(actual.reshape(-1, 1)).ravel()
    
    residuals = predicted - actual
    std_res = np.std(residuals)
    n = len(actual)
    
    # Calculate confidence intervals
    if n > 2:
        standard_error = std_res * np.sqrt(1 + 1/n + (actual - np.mean(actual))**2 / np.sum((actual - np.mean(actual))**2))
        t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence
        upper_bound = y_fit + t_val * standard_error
        lower_bound = y_fit - t_val * standard_error
    else:
        upper_bound = y_fit
        lower_bound = y_fit
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=actual, y=predicted, edgecolor='k', alpha=0.7, s=60)
    plt.plot(actual, y_fit, color='red', linewidth=2, 
              label=f'Fit: y={slope:.3f}x + {intercept:.3f}\n(SD slope: {slope_sd:.3f}, SD intercept: {intercept_sd:.3f})')
    
    
    
    # Annotate outliers
    # if n > 0:
    #     for i, (x_val, y_val, lb, ub) in enumerate(zip(actual, predicted, lower_bound, upper_bound)): 
    #         if y_val > ub or y_val < lb: 
    #             plt.annotate(str(label[i]), (x_val, y_val), 
    #                         textcoords="offset points", xytext=(5, 5), 
    #                         ha='center', fontsize=8, alpha=0.7)
    
    if n > 2:
        plt.fill_between(actual, lower_bound, upper_bound, color='red', alpha=0.2, label='95% Confidence Interval')
    
        
    
    plt.title(f' ({dataset_name} )', fontsize=16)
    plt.xlabel('Experimental pk', fontsize=14)
    plt.ylabel('Calculated pk', fontsize=14)
    # plt.legend(fontsize=14, loc='lower right')
    plt.legend(
    fontsize=14,
    loc='lower right',
    frameon=True,
    borderpad=1.2,
    labelspacing=0.8,
    handlelength=3
)
    plt.grid(True, alpha=0.3)
    
    # min_val = min(actual.min() , predicted.min())-0.2
    # max_val = max(actual.max(), predicted.max())+0.2
    min_val = min(actual.min() , predicted.min())-0.2
    max_val = 3.5
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.savefig(f'calculated_vs_experimental_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
 


def compute_regression_stats(actual, predicted, num_iterations=1000):
    """
    Computes the slope and intercept statistics using bootstrapping.
    """
    if len(actual) < 10:
        # Use simple regression for small datasets
        reg = LinearRegression().fit(actual.reshape(-1, 1), predicted)
        slope_mean = reg.coef_[0]
        intercept_mean = reg.intercept_
        slope_sd = 0.0
        intercept_sd = 0.0
    else:
        slopes = []
        intercepts = []
        for _ in range(num_iterations):
            indices = np.random.choice(len(actual), len(actual), replace=True)
            sample_actual = actual[indices].reshape(-1, 1)
            sample_predicted = predicted[indices].reshape(-1, 1)
            reg = LinearRegression().fit(sample_actual, sample_predicted)
            slopes.append(reg.coef_[0][0])
            intercepts.append(reg.intercept_[0])
        
        slope_mean = np.mean(slopes)
        intercept_mean = np.mean(intercepts)
        slope_sd = np.std(slopes)
        intercept_sd = np.std(intercepts)
    
    # Calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    prediction_results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'slope': slope_mean,
        'intercept': intercept_mean,
        'slope_sd': slope_sd,
        'intercept_sd': intercept_sd
    }
    
    return slope_mean, intercept_mean, slope_sd, intercept_sd, prediction_results

def parity_plot_from_collect(out, title):
    y_pred = out[2].reshape(-1, 1)
    y_true = out[3].reshape(-1, 1)

    lr = LinearRegression().fit(y_true, y_pred)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot(y_true, y_true, '--', label="Ideal")
    plt.plot(y_true, slope*y_true + intercept, label="Fit")
    plt.xlabel("Experimental")
    plt.ylabel("Predicted")
    plt.title(f"{title}\nSlope={slope:.2f}, Intercept={intercept:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_combined_williams(train, val, test,
                           train_pred, val_pred, test_pred,
                           train_nom, val_nom, test_nom,
                           train_idx, val_idx, test_idx):
    """
    plot all three data sets on one williams plot
    """
    def notation(h, h_crit, std_residuals, idx):
        for i in range(len(h)):
            if h[i] > h_crit or std_residuals[i] > 3 or std_residuals[i] < -3:
                plt.annotate(idx[i], (h[i], std_residuals[i]),
                             textcoords="offset points", xytext=(5,5),
                             ha='center', fontsize=12)
              
    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)      
    test_scaled = scaler.transform(test)
    
    pca = PCA(n_components=0.98, svd_solver='full')
    pca.fit(train_scaled)
    train_scores = pca.transform(train_scaled)   
    val_scores = pca.transform(val_scaled)
    test_scores = pca.transform(test_scaled)
    

    try:
        scores_dict = {'train': train_scores, 'val': val_scores, 'test': test_scores}
        train_inv = np.linalg.inv(train_scores.T @ train_scores)
        leverage_dict = {}
        for dset, scores in scores_dict.items():
            leverage_dict[dset] = np.diag(scores @ train_inv @ scores.T)
        train_lev = leverage_dict['train']
        val_lev   = leverage_dict['val']
        test_lev  = leverage_dict['test']
    except Exception as e:
        print(f'Error in leverage calculation: {e}')
        return
    
    # محاسبه residual استاندارد
    try:
        pred_dict = {
            'train': {'pred': train_pred, 'nom': train_nom},
            'val':   {'pred': val_pred,   'nom': val_nom},
            'test':  {'pred': test_pred,  'nom': test_nom}
        }
        std_residual_dict = {}
        for name, data in pred_dict.items():
            residual = data['pred'] - data['nom']
            std_residual_dict[name] = (residual - np.mean(residual)) / np.std(residual)
        train_std_residuals = std_residual_dict['train']
        val_std_residuals   = std_residual_dict['val']
        test_std_residuals  = std_residual_dict['test']
    except Exception as e:
        print(f'Error in std residual calculation: {e}')
        return
        
    # آستانه leverage
    p = train_scores.shape[1]
    n = len(train_pred)
    h_crit = 3 * p / n
    
    plt.figure(figsize=(10,8))
    plt.scatter(train_lev, train_std_residuals, label='Train', edgecolors='k', color='blue', s=60)
    plt.scatter(val_lev,   val_std_residuals,   label='Validation', edgecolors='k', color='green', s=60)
    plt.scatter(test_lev,  test_std_residuals,  label='Test', marker='D', edgecolors='k', color='orange', s=60)
    
    notation(train_lev, h_crit, train_std_residuals, train_idx)
    notation(val_lev,   h_crit, val_std_residuals,   val_idx)
    notation(test_lev,  h_crit, test_std_residuals,  test_idx)
    
    plt.axhline(y=3, color='r', linestyle='--')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.axvline(x=h_crit, color='g', linestyle='--')
    plt.annotate(f' h* = {h_crit:.2f}', (h_crit, np.min(train_std_residuals)), fontsize=14)
    plt.legend(fontsize=14, loc='upper right')
    plt.xlabel('Leverage', fontsize=14)
    plt.ylabel('Std residuals', fontsize=14)
    plt.title('Williams plot', fontsize=16)
    plt.savefig('Williams_Plot_Combined.png', dpi=300)
    plt.show()
    
def plot_residuals(y_true, y_pred, j, scaler=None):
    residuals = y_pred - y_true
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)  # پیش‌فرض ddof=0

    if std_res == 0:
        raise ValueError('Standard deviation is zero.')

    residuals_scaled = (residuals - mean_res) / std_res

    plt.figure(figsize=(6, 4))

    count, bins, patches = plt.hist(
        residuals_scaled, bins=25, edgecolor='k', alpha=0.7, density=True,
        label='Residuals (scaled)'
    )

    plt.axvline(0, linestyle='--', linewidth=1, color='gray', label='Zero residual')

    x = np.linspace(bins[0], bins[-1], 200)
    pdf = norm.pdf(x)  

    plt.plot(x, pdf, 'k:', linewidth=1.5, label='Normal (μ=0, σ=1)',color='red')

    textstr = f'μ = {mean_res:.3f}\nσ = {std_res:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=props)

    plt.xlabel("Residual (Predicted − Experimental)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    title_map = {
        'testset': "Residual Distribution (Test Set)",
        'trainset': "Residual Distribution (Training Set)",
        'valset': "Residual Distribution (Validation Set)"
    }
    plt.title(title_map.get(j, "Residual Distribution"), fontsize=13)

    plt.legend(loc='upper right')
    plt.tight_layout()

    filename_map = {
        'testset': "Residual_Distribution_test_set.png",
        'trainset': "Residual_Distribution_train_set.png",
        'valset': "Residual_Distribution_validation_set.png"
    }
    plt.savefig(filename_map.get(j, "Residual_Distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()

def calculate_kunal_roy_validation(y_experimental, y_predicted):
    y_obs = np.asarray(y_experimental).flatten().astype(float)
    y_pred = np.asarray(y_predicted).flatten().astype(float)

    if len(y_obs) != len(y_pred):
        raise ValueError("Length mismatch: y_experimental and y_predicted must have same length.")
    if len(y_obs) < 3:
        raise ValueError("At least 3 samples required.")

    def compute_r2m_single_direction(y1, y2):
        if np.std(y1) == 0 or np.std(y2) == 0:
            return 0.0
        r = np.corrcoef(y1, y2)[0, 1]
        r2 = r ** 2
        k = np.sum(y1 * y2) / np.sum(y2 ** 2)
        y_ro = k * y2
        ss_res = np.sum((y1 - y_ro) ** 2)
        ss_tot = np.sum((y1 - np.mean(y1)) ** 2)
        r02 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
        diff = r2 - r02
        if diff < 0:
            return 0.0
        sqrt_term = np.sqrt(diff)
        r2m = r2 * (1 - sqrt_term)
        return max(0.0, r2m)

    r2m_forward = compute_r2m_single_direction(y_obs, y_pred)
    r2m_reverse = compute_r2m_single_direction(y_pred, y_obs)
    r2m_avg = (r2m_forward + r2m_reverse) / 2
    delta_r2m = abs(r2m_forward - r2m_reverse)

    r2_std = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))

    passed = (r2m_avg > 0.5) and (delta_r2m < 0.2)

    return {
        "Standard R²": round(r2_std, 4),
        "RMSE": round(rmse, 4),
        "Average r²_m": round(r2m_avg, 4),
        "Delta r²_m": round(delta_r2m, 4),
        "Roy's Criteria": "PASSED" if passed else "FAILED"
    }

def parity_plot_publication(y_true, y_pred, title, fname):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=45, edgecolor='k', alpha=0.7)
    
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    plt.plot(lims, lims, 'k--', linewidth=1.5)

    plt.xlabel("Experimental pk", fontsize=12)
    plt.ylabel("Predicted pk", fontsize=12)
    plt.title(title, fontsize=13)

    plt.text(
        0.05, 0.95,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()

def plot_error_vs_k(y_true, y_pred):
    errors = np.abs(y_pred - y_true)

    plt.figure(figsize=(5,4))
    plt.scatter(y_true, errors, alpha=0.7, edgecolor='k')
    plt.xlabel("Experimental pk", fontsize=12)
    plt.ylabel("|Prediction Error|", fontsize=12)
    plt.title("Error vs Experimental pk", fontsize=13)
    plt.tight_layout()
    plt.savefig("Error_vs_k.png", dpi=300)
    plt.show()

def plot_pca_combined(train, test):
    pca = PCA(n_components=2)
    X = np.vstack([train, test])
    X_pca = pca.fit_transform(X)

    n_train = len(train)

    plt.figure(figsize=(5,5))
    plt.scatter(X_pca[:n_train,0], X_pca[:n_train,1],
                label="Train", alpha=0.7)
    plt.scatter(X_pca[n_train:,0], X_pca[n_train:,1],
                label="Test", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("PCA of Combined Features")
    plt.tight_layout()
    plt.savefig("PCA_Combined.png", dpi=300)
    plt.show()

# -----------------------------  PCA/UMAP ----------------------------- #
def extract_all_features(model, device, loader, target_scaler, original_df, split_name="Test"):
    model.eval()
    
    features = {
        'combined': [], 'mol_nodes': [], 'crystal_nodes': [],
        'mol_global': [], 'crystal_global': [], 'experimental': []
    }
    targets_list = []
    indices_list = []
    splits_list = []
    
    with torch.no_grad():
        for mol_graphs, crystal_graphs, exp_feats, targets, batch_indices in loader:  
            mol_graphs = mol_graphs.to(device)
            crystal_graphs = crystal_graphs.to(device)
            exp_feats = exp_feats.to(device)
            batch_indices = batch_indices.cpu().numpy()
            
            _, combined = model(mol_graphs, crystal_graphs, exp_feats)
            
            x_mol = F.relu(model.mol_gn1(
                model.mol_conv1(mol_graphs.x, mol_graphs.edge_index, mol_graphs.edge_attr),
                mol_graphs.batch
            ))
            x_mol = F.relu(model.mol_gn2(
                model.mol_conv2(x_mol, mol_graphs.edge_index, mol_graphs.edge_attr),
                mol_graphs.batch
            ))
            x_mol = F.relu(model.mol_gn3(
                model.mol_conv3(x_mol, mol_graphs.edge_index, mol_graphs.edge_attr),
                mol_graphs.batch
            ))
            
            x_crystal = F.relu(model.crystal_gn1(
                model.crystal_conv1(
                    crystal_graphs.x,
                    crystal_graphs.edge_index,
                    crystal_graphs.edge_attr
                ),
                crystal_graphs.batch
            ))
            x_crystal = F.relu(model.crystal_gn2(
                model.crystal_conv2(
                    x_crystal,
                    crystal_graphs.edge_index,
                    crystal_graphs.edge_attr
                ),
                crystal_graphs.batch
            ))
            x_crystal = F.relu(model.crystal_gn3(
                model.crystal_conv3(
                    x_crystal,
                    crystal_graphs.edge_index,
                    crystal_graphs.edge_attr
                ),
                crystal_graphs.batch
            ))
            
            n_samples = mol_graphs.batch.max().item() + 1
            for i in range(n_samples):
                features['combined'].append(combined[i].cpu().numpy())
                
                mol_mask = mol_graphs.batch == i
                mol_node_feat = x_mol[mol_mask].mean(dim=0).cpu().numpy()
                features['mol_nodes'].append(mol_node_feat)
                
                crystal_mask = crystal_graphs.batch == i
                crystal_node_feat = x_crystal[crystal_mask].mean(dim=0).cpu().numpy()
                features['crystal_nodes'].append(crystal_node_feat)
                
                if hasattr(mol_graphs, 'mol_global_features'):
                    mol_global = mol_graphs.mol_global_features[i].cpu().numpy()
                    features['mol_global'].append(mol_global)
                if hasattr(crystal_graphs, 'global_features'):
                    crystal_global = crystal_graphs.global_features[i].cpu().numpy()
                    features['crystal_global'].append(crystal_global)
                    
                features['experimental'].append(exp_feats[i].cpu().numpy())
                targets_list.append(targets[i].cpu().numpy())
                indices_list.append(batch_indices[i])      
                splits_list.append(split_name)
    
    for key in features:
        if features[key]:
            features[key] = np.vstack(features[key])
            print(f"   • {key:15s}: {features[key].shape}")
    
    targets = np.vstack(targets_list)
    if target_scaler is not None:
        targets_real = target_scaler.inverse_transform(targets).flatten()
    else:
        targets_real = targets.flatten()
    
    return features, targets_real, np.array(indices_list), splits_list

def create_pca_umap_plots(features, targets, feature_name, indices, original_df, experimental_cols):
    print(f"\n{'='*50}")
    print(f" Analyzing: {feature_name}")
    print(f"{'='*50}")
    print(f"   Feature shape: {features.shape}")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    hover_texts = []
    for idx in indices:
        row = original_df.iloc[idx]
        hover_text = (
            f"<b>Index: {idx}</b><br>"
            f"────────────────<br>"
            f"<b> pk value:</b> {row['k']:.4f}<br>"
            f"<b> Photocatalyst:</b> {row['Photocat']}<br>"
            f"<b> SMILES:</b> {row['smiles']}<br>"
            f"────────────────<br>"
            f"<b> Experimental Conditions:</b><br>"
        )
        for exp_col in experimental_cols:
            if exp_col in row.index:
                if 'dosage' in exp_col.lower():
                    display_name = 'Dosage (g/L)'
                elif 'concentration' in exp_col.lower():
                    display_name = 'Concentration (ppm)'
                elif 'ph' in exp_col.lower():
                    display_name = 'pH'
                elif 'light' in exp_col.lower():
                    display_name = 'Light Intensity'
                else:
                    display_name = exp_col
                hover_text += f"   • {display_name}: {row[exp_col]}<br>"
        hover_texts.append(hover_text)
    
    print("\n    Performing PCA...")
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(features_scaled)
    print(f"      → {features_pca.shape[1]} components selected")
    print(f"      → Total variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    clean_name = feature_name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('&', 'and')
    
    # PCA 2D
    fig_pca_2d = go.Figure()
    fig_pca_2d.add_trace(go.Scatter(
        x=features_pca[:, 0], y=features_pca[:, 1],
        mode='markers',
        marker=dict(size=8, color=targets, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='pk value', x=1.02, titleside='right'),
                    line=dict(width=0.5, color='black')),
        text=hover_texts, hoverinfo='text', name='PCA'
    ))
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    total_var_2d = var_pc1 + var_pc2
    fig_pca_2d.update_layout(
        title=dict(text=f'<b>PCA - {feature_name}</b><br><sup>PC1: {var_pc1:.2f}% | PC2: {var_pc2:.2f}% | Total: {total_var_2d:.2f}%</sup>',
                  font=dict(size=14), x=0.5, xanchor='center'),
        xaxis=dict(title=f'PC1 ({var_pc1:.2f}%)', gridcolor='lightgray', gridwidth=1),
        yaxis=dict(title=f'PC2 ({var_pc2:.2f}%)', gridcolor='lightgray', gridwidth=1),
        plot_bgcolor='white', width=900, height=700,
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    fig_pca_2d.write_html(f'pca_2d_{clean_name}_with_info.html')
    print(f"      Saved: pca_2d_{clean_name}_with_info.html")
    
    # PCA 3D
    fig_pca_3d = go.Figure()
    fig_pca_3d.add_trace(go.Scatter3d(
        x=features_pca[:, 0], y=features_pca[:, 1],
        z=features_pca[:, 2] if features_pca.shape[1] >= 3 else features_pca[:, 0],
        mode='markers',
        marker=dict(size=6, color=targets, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='pk value', x=1.02), line=dict(width=0.5, color='black')),
        text=hover_texts, hoverinfo='text'
    ))
    var_pc3 = pca.explained_variance_ratio_[2] * 100 if features_pca.shape[1] >= 3 else var_pc1
    total_var_3d = var_pc1 + var_pc2 + var_pc3
    fig_pca_3d.update_layout(
        title=dict(text=f'<b>PCA 3D - {feature_name}</b><br><sup>Total variance (3 PCs): {total_var_3d:.2f}%</sup>',
                  font=dict(size=14), x=0.5, xanchor='center'),
        scene=dict(xaxis=dict(title=f'PC1 ({var_pc1:.2f}%)', gridcolor='lightgray'),
                   yaxis=dict(title=f'PC2 ({var_pc2:.2f}%)', gridcolor='lightgray'),
                   zaxis=dict(title=f'PC3 ({var_pc3:.2f}%)' if features_pca.shape[1] >= 3 else 'PC1', gridcolor='lightgray'),
                   bgcolor='white', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)), aspectmode='cube'),
        width=1000, height=800,
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    fig_pca_3d.write_html(f'pca_3d_{clean_name}_with_info.html')
    print(f"      ✅ Saved: pca_3d_{clean_name}_with_info.html")
    
    # UMAP
    print("\n    Performing UMAP with different configurations...")
    umap_configs = [
        {'n_neighbors': 15, 'min_dist': 0.1, 'desc': 'Local Structure'},
        {'n_neighbors': 30, 'min_dist': 0.5, 'desc': 'Global Structure'},
        {'n_neighbors': 50, 'min_dist': 0.1, 'desc': 'Balanced'}
    ]
    for i, config in enumerate(umap_configs):
        print(f"      • UMAP {i+1}: {config['desc']} (n_neighbors={config['n_neighbors']}, min_dist={config['min_dist']})")
        umap = UMAP(n_neighbors=config['n_neighbors'], min_dist=config['min_dist'],
                    n_components=3, random_state=42, n_epochs=200)
        features_umap = umap.fit_transform(features_scaled)
        
        fig_umap = go.Figure()
        fig_umap.add_trace(go.Scatter3d(
            x=features_umap[:, 0], y=features_umap[:, 1], z=features_umap[:, 2],
            mode='markers',
            marker=dict(size=6, color=targets, colorscale='Viridis', showscale=True,
                        colorbar=dict(title='pk value', x=1.02), line=dict(width=0.5, color='black')),
            text=hover_texts, hoverinfo='text'
        ))
        fig_umap.update_layout(
            title=dict(text=f'<b>UMAP 3D - {feature_name}</b><br><sup>{config["desc"]} | n_neighbors={config["n_neighbors"]}, min_dist={config["min_dist"]}</sup>',
                      font=dict(size=14), x=0.5, xanchor='center'),
            scene=dict(xaxis=dict(title='Components_1'), yaxis=dict(title='Components_2'), zaxis=dict(title='Components_3'),
                       bgcolor='white', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)), aspectmode='cube'),
            width=1000, height=800,
            hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        fig_umap.write_html(f'umap_3d_{clean_name}_config{i+1}_with_info.html')
        print(f"          Saved: umap_3d_{clean_name}_config{i+1}_with_info.html")
        
        fig_umap_2d = go.Figure()
        fig_umap_2d.add_trace(go.Scatter(
            x=features_umap[:, 0], y=features_umap[:, 1], mode='markers',
            marker=dict(size=8, color=targets, colorscale='Viridis', showscale=True,
                        colorbar=dict(title='pk value', x=1.02), line=dict(width=0.5, color='black')),
            text=hover_texts, hoverinfo='text'
        ))
        fig_umap_2d.update_layout(
            title=dict(text=f'<b>UMAP 2D - {feature_name}</b><br><sup>{config["desc"]} | n_neighbors={config["n_neighbors"]}, min_dist={config["min_dist"]}</sup>',
                      font=dict(size=14), x=0.5, xanchor='center'),
            xaxis=dict(title='UMAP1', gridcolor='lightgray'), yaxis=dict(title='UMAP2', gridcolor='lightgray'),
            plot_bgcolor='white', width=900, height=700,
            hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
            margin=dict(l=80, r=80, t=100, b=80)
        )
        fig_umap_2d.write_html(f'umap_2d_{clean_name}_config{i+1}_with_info.html')
        print(f"          Saved: umap_2d_{clean_name}_config{i+1}_with_info.html")
    
    # Variance plot
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        y=pca.explained_variance_ratio_ * 100,
        name='Individual variance', marker_color='steelblue',
        text=[f'{var*100:.2f}%' for var in pca.explained_variance_ratio_], textposition='outside'
    ))
    fig_var.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        y=np.cumsum(pca.explained_variance_ratio_) * 100,
        name='Cumulative variance', mode='lines+markers+text',
        marker=dict(color='red', size=8), line=dict(color='red', width=2, dash='dash'),
        text=[f'{cum*100:.1f}%' for cum in np.cumsum(pca.explained_variance_ratio_)], textposition='top center'
    ))
    fig_var.add_hline(y=95, line_dash="dash", line_color="green", line_width=2,
                      annotation_text="95% variance", annotation_position="bottom right")
    fig_var.update_layout(
        title=dict(text=f'<b>PCA Variance Analysis - {feature_name}</b><br><sup>Components: {features_pca.shape[1]} | Total variance: {pca.explained_variance_ratio_.sum():.2%}</sup>',
                  font=dict(size=14), x=0.5, xanchor='center'),
        xaxis_title='Principal Components', yaxis_title='Variance (%)',
        width=1000, height=600, yaxis=dict(range=[0, 105], ticksuffix='%'),
        margin=dict(l=80, r=50, t=100, b=80)
    )
    fig_var.write_html(f'pca_variance_{clean_name}.html')
    print(f"       Saved: pca_variance_{clean_name}.html")
    
    print(f"\n    All plots for {feature_name} completed!")
    return {'pca_2d': fig_pca_2d, 'pca_3d': fig_pca_3d, 'pca_variance': fig_var}

def plot_pca_umap_with_splits(feature_type, features_dict_train, features_dict_val, features_dict_test,
                              targets_train, targets_val, targets_test, 
                              indices_train, indices_val, indices_test,
                              original_df, experimental_cols):
    print(f"\n{'='*50}")
    print(f" Analyzing: {feature_type}")
    print(f"{'='*50}")
    
    all_features = np.vstack([
        features_dict_train[feature_type],
        features_dict_val[feature_type],
        features_dict_test[feature_type]
    ])
    all_targets = np.concatenate([targets_train, targets_val, targets_test])
    all_indices = np.concatenate([indices_train, indices_val, indices_test])
    all_splits = ['Train'] * len(features_dict_train[feature_type]) + \
                  ['Validation'] * len(features_dict_val[feature_type]) + \
                  ['Test'] * len(features_dict_test[feature_type])
    
    print(f"   Total samples: {len(all_features)}")
    print(f"   Features dimension: {all_features.shape[1]}")
    
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    hover_texts = []
    for idx, split in zip(all_indices, all_splits):
        row = original_df.iloc[idx]
        hover_text = (
            f"<b>Index: {idx}</b><br>"
            f"<b>Split: {split}</b><br>────────────────<br>"
            f"<b> pk value:</b> {row['k']:.4f}<br>"
            f"<b> Photocatalyst:</b> {row['Photocat']}<br>"
            f"<b> SMILES:</b> {row['smiles']}<br>────────────────<br>"
            f"<b> Experimental Conditions:</b><br>"
        )
        for exp_col in experimental_cols:
            if exp_col in row.index:
                if 'dosage' in exp_col.lower():
                    display_name = 'Dosage (g/L)'
                elif 'concentration' in exp_col.lower():
                    display_name = 'Concentration (ppm)'
                elif 'ph' in exp_col.lower():
                    display_name = 'pH'
                elif 'light' in exp_col.lower():
                    display_name = 'Light Intensity'
                else:
                    display_name = exp_col
                hover_text += f"   • {display_name}: {row[exp_col]}<br>"
        hover_texts.append(hover_text)
    
    colors = {'Train': 'blue', 'Validation': 'green', 'Test': 'orange'}
    
    print("\n    Performing PCA...")
    pca = PCA(n_components=0.95)
    all_features_pca = pca.fit_transform(all_features_scaled)
    print(f"      → {all_features_pca.shape[1]} components selected")
    print(f"      → Total variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    clean_name = feature_type.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    
    # PCA 2D with splits
    fig_pca_2d = go.Figure()
    for split in ['Train', 'Validation', 'Test']:
        mask = np.array(all_splits) == split
        fig_pca_2d.add_trace(go.Scatter(
            x=all_features_pca[mask, 0], y=all_features_pca[mask, 1], mode='markers', name=split,
            marker=dict(size=8, color=colors[split], line=dict(width=0.5, color='black'),
                        symbol='circle' if split == 'Train' else 'square' if split == 'Validation' else 'diamond'),
            text=[hover_texts[i] for i in range(len(all_splits)) if all_splits[i] == split], hoverinfo='text'
        ))
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    fig_pca_2d.update_layout(
        title=dict(text=f'<b>PCA - {feature_type}</b><br><sup>PC1: {var_pc1:.2f}% | PC2: {var_pc2:.2f}% | Total: {var_pc1 + var_pc2:.2f}%</sup>',
                  font=dict(size=14), x=0.5),
        xaxis=dict(title=f'PC1 ({var_pc1:.2f}%)', gridcolor='lightgray'),
        yaxis=dict(title=f'PC2 ({var_pc2:.2f}%)', gridcolor='lightgray'),
        plot_bgcolor='white', width=1000, height=700,
        hoverlabel=dict(bgcolor='white', font_size=11),
        legend=dict(title='Dataset Split', x=0.85, y=0.95, bgcolor='rgba(255,255,255,0.8)')
    )
    fig_pca_2d.write_html(f'pca_2d_{clean_name}_with_splits.html')
    print(f"       Saved: pca_2d_{clean_name}_with_splits.html")
    
    # PCA 3D with splits
    fig_pca_3d = go.Figure()
    for split in ['Train', 'Validation', 'Test']:
        mask = np.array(all_splits) == split
        fig_pca_3d.add_trace(go.Scatter3d(
            x=all_features_pca[mask, 0], y=all_features_pca[mask, 1],
            z=all_features_pca[mask, 2] if all_features_pca.shape[1] >= 3 else all_features_pca[mask, 0],
            mode='markers', name=split,
            marker=dict(size=6, color=colors[split], line=dict(width=0.5, color='black'), symbol='circle'),
            text=[hover_texts[i] for i in range(len(all_splits)) if all_splits[i] == split], hoverinfo='text'
        ))
    var_pc3 = pca.explained_variance_ratio_[2] * 100 if all_features_pca.shape[1] >= 3 else var_pc1
    fig_pca_3d.update_layout(
        title=dict(text=f'<b>PCA 3D - {feature_type}</b><br><sup>Variance: PC1={var_pc1:.2f}%, PC2={var_pc2:.2f}%, PC3={var_pc3:.2f}%</sup>',
                  font=dict(size=14), x=0.5),
        scene=dict(xaxis=dict(title=f'PC1 ({var_pc1:.2f}%)'), yaxis=dict(title=f'PC2 ({var_pc2:.2f}%)'),
                   zaxis=dict(title=f'PC3 ({var_pc3:.2f}%)' if all_features_pca.shape[1] >= 3 else 'PC1'), bgcolor='white'),
        width=1000, height=800, legend=dict(title='Dataset Split', x=0.85, y=0.95)
    )
    fig_pca_3d.write_html(f'pca_3d_{clean_name}_with_splits.html')
    print(f"      Saved: pca_3d_{clean_name}_with_splits.html")
    
    # UMAP with splits
    print("\n    Performing UMAP with different configurations...")
    umap_configs = [
        {'n_neighbors': 15, 'min_dist': 0.1, 'desc': 'Local Structure'},
        {'n_neighbors': 30, 'min_dist': 0.5, 'desc': 'Global Structure'},
        {'n_neighbors': 50, 'min_dist': 0.1, 'desc': 'Balanced'}
    ]
    for i, config in enumerate(umap_configs):
        print(f"      • UMAP {i+1}: {config['desc']}")
        umap = UMAP(n_neighbors=config['n_neighbors'], min_dist=config['min_dist'], n_components=3, random_state=42)
        all_features_umap = umap.fit_transform(all_features_scaled)
        
        fig_umap = go.Figure()
        for split in ['Train', 'Validation', 'Test']:
            mask = np.array(all_splits) == split
            fig_umap.add_trace(go.Scatter3d(
                x=all_features_umap[mask, 0], y=all_features_umap[mask, 1], z=all_features_umap[mask, 2],
                mode='markers', name=split,
                marker=dict(size=5, color=colors[split], line=dict(width=0.5, color='black')),
                text=[hover_texts[i] for i in range(len(all_splits)) if all_splits[i] == split], hoverinfo='text'
            ))
        fig_umap.update_layout(
            title=dict(text=f'<b>UMAP 3D - {feature_type}</b><br><sup>{config["desc"]}</sup>', font=dict(size=14), x=0.5),
            scene=dict(xaxis=dict(title='Components_1'), yaxis=dict(title='Components_2'), zaxis=dict(title='Components_3'), bgcolor='white'),
            width=1000, height=800, legend=dict(title='Dataset Split')
        )
        fig_umap.write_html(f'umap_3d_{clean_name}_config{i+1}_with_splits.html')
        print(f"          Saved: umap_3d_{clean_name}_config{i+1}_with_splits.html")
        
        fig_umap_2d = go.Figure()
        for split in ['Train', 'Validation', 'Test']:
            mask = np.array(all_splits) == split
            fig_umap_2d.add_trace(go.Scatter(
                x=all_features_umap[mask, 0], y=all_features_umap[mask, 1], mode='markers', name=split,
                marker=dict(size=7, color=colors[split], line=dict(width=0.5, color='black')),
                text=[hover_texts[i] for i in range(len(all_splits)) if all_splits[i] == split], hoverinfo='text'
            ))
        fig_umap_2d.update_layout(
            title=dict(text=f'<b>UMAP 2D - {feature_type}</b><br><sup>{config["desc"]}</sup>', font=dict(size=14), x=0.5),
            xaxis=dict(title='UMAP1', gridcolor='lightgray'), yaxis=dict(title='UMAP2', gridcolor='lightgray'),
            plot_bgcolor='white', width=900, height=700, legend=dict(title='Dataset Split')
        )
        fig_umap_2d.write_html(f'umap_2d_{clean_name}_config{i+1}_with_splits.html')
        print(f"          Saved: umap_2d_{clean_name}_config{i+1}_with_splits.html")
    
    return fig_pca_2d, fig_pca_3d

def plot_model_space_3d(features_dict_train, features_dict_val, features_dict_test,
                        targets_train, targets_val, targets_test,
                        indices_train, indices_val, indices_test,
                        splits_train, splits_val, splits_test,
                        original_df, feature_type="combined", numerical_features=[]):
    print(f"\n{'='*50}")
    print(f" Analyzing: {feature_type}")
    print(f"{'='*50}")

    X = np.vstack([
        features_dict_train[feature_type],
        features_dict_val[feature_type],
        features_dict_test[feature_type]
    ])
    targets = np.concatenate([targets_train, targets_val, targets_test])
    all_indices = np.concatenate([indices_train, indices_val, indices_test])
    all_splits = np.concatenate([splits_train, splits_val, splits_test])

    required_cols = ['Photocat', 'smiles', 'k']
    missing_cols = [col for col in required_cols if col not in original_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}\nAvailable: {list(original_df.columns)}")

    hover_texts = []
    for i, idx in enumerate(all_indices):
        row = original_df.iloc[idx]
        split = all_splits[i]
        k_val = row.get('k')
        k_str = f"{k_val:.6f}" if pd.notna(k_val) else "N/A"
        photocat = row.get('Photocat')
        photocat_str = str(photocat) if pd.notna(photocat) else "N/A"
        hover_text = (
            f"<b>Index: {idx} | Split: {split}</b><br>────────────────<br>"
            f"<b> pk value:</b> {k_str}<br>"
            f"<b> Photocatalyst:</b> {photocat_str}<br>────────────────<br>"
            f"<b> Experimental Conditions:</b><br>"
        )
        for exp_col in numerical_features:
            if exp_col in row.index and pd.notna(row.get(exp_col)):
                exp_lower = exp_col.lower()
                if 'dosage' in exp_lower:
                    display_name = 'Dosage (g/L)'
                elif 'concentration' in exp_lower:
                    display_name = 'Concentration (ppm)'
                elif 'ph' in exp_lower:
                    display_name = 'pH'
                elif 'light' in exp_lower:
                    display_name = 'Light Intensity'
                else:
                    display_name = exp_col
                val = row.get(exp_col)
                val_str = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                hover_text += f"   • {display_name}: {val_str}<br>"
        hover_texts.append(hover_text)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n    Performing PCA (0.98 variance)...")
    pca = PCA(n_components=0.98, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    n_components = X_pca.shape[1]
    explained_ratios = pca.explained_variance_ratio_
    total_var = explained_ratios.sum()

    if n_components < 3:
        print(f" Only {n_components} PCs needed for 98% variance. Padding to 3D.")
        X_pca = np.pad(X_pca, ((0, 0), (0, 3 - n_components)), mode='constant')
        var_pc1, var_pc2, var_pc3 = explained_ratios[0], explained_ratios[1], 0.0
    else:
        var_pc1, var_pc2, var_pc3 = explained_ratios[0], explained_ratios[1], explained_ratios[2]

    x_pca, y_pca, z_pca = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

    fig_pca = go.Figure()
    fig_pca.add_trace(go.Scatter3d(
        x=x_pca, y=y_pca, z=z_pca, mode='markers',
        marker=dict(size=6, color=targets, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='pk', x=1.02, titleside='right', tickformat='.4f'),
                    line=dict(width=0.5, color='black')),
        text=hover_texts, hoverinfo='text'
    ))
    fig_pca.update_layout(
        title=dict(text=f'<b>PCA 3D — {feature_type}</b><br><sup>Total Variance: {total_var:.1%} | PC1: {var_pc1:.1%} | PC2: {var_pc2:.1%} | PC3: {var_pc3:.1%}</sup>',
                  font=dict(size=14), x=0.5),
        scene=dict(xaxis=dict(title=f'PC1 ({var_pc1:.1%})'), yaxis=dict(title=f'PC2 ({var_pc2:.1%})'),
                   zaxis=dict(title=f'PC3 ({var_pc3:.1%})'), bgcolor='white'),
        width=1000, height=800, hoverlabel=dict(bgcolor='white', font_size=11), margin=dict(l=50, r=50, t=100, b=50)
    )
    clean_name = feature_type.lower().replace(' ', '_')
    fig_pca.write_html(f'pca_3d_{clean_name}_k_color.html')
    print(f"       Saved: pca_3d_{clean_name}_k_color.html")

    print("\n    Performing UMAP (n_neighbors=30, min_dist=0.5)...")
    reducer = UMAP(n_components=3, n_neighbors=30, min_dist=0.5, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    x_umap, y_umap, z_umap = X_umap[:, 0], X_umap[:, 1], X_umap[:, 2]
    fig_umap = go.Figure()
    fig_umap.add_trace(go.Scatter3d(
        x=x_umap, y=y_umap, z=z_umap, mode='markers',
        marker=dict(size=6, color=targets, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='pk', x=1.02, titleside='right', tickformat='.4f'),
                    line=dict(width=0.5, color='black')),
        text=hover_texts, hoverinfo='text'
    ))
    fig_umap.update_layout(
        title=dict(text=f'<b>UMAP 3D — {feature_type}</b><br><sup>n_neighbors=30, min_dist=0.5</sup>',
                  font=dict(size=14), x=0.5),
        scene=dict(xaxis=dict(title='Componenets_1'), yaxis=dict(title='Componenets_2'), zaxis=dict(title='Componenets_3'), bgcolor='white'),
        width=1000, height=800, hoverlabel=dict(bgcolor='white', font_size=11), margin=dict(l=50, r=50, t=100, b=50)
    )
    fig_umap.write_html(f'umap_3d_{clean_name}_k_color.html')
    print(f"      Saved: umap_3d_{clean_name}_k_color.html")

def plot_tsne_3d(features_dict_train, features_dict_val, features_dict_test,
                 targets_train, targets_val, targets_test,
                 indices_train, indices_val, indices_test,
                 splits_train, splits_val, splits_test,
                 original_df, feature_type="combined", numerical_features=[]):
    print(f"\n{'='*50}")
    print(f" Running t-SNE on: {feature_type}")
    print(f"{'='*50}")

    X = np.vstack([
        features_dict_train[feature_type],
        features_dict_val[feature_type],
        features_dict_test[feature_type]
    ])
    targets = np.concatenate([targets_train, targets_val, targets_test])
    all_indices = np.concatenate([indices_train, indices_val, indices_test])
    all_splits = np.concatenate([splits_train, splits_val, splits_test])

    required_cols = ['Photocat', 'smiles', 'k']
    missing_cols = [col for col in required_cols if col not in original_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    hover_texts = []
    for i, idx in enumerate(all_indices):
        row = original_df.iloc[idx]
        split = all_splits[i]
        k_val = row.get('k')
        k_str = f"{k_val:.6f}" if pd.notna(k_val) else "N/A"
        photocat = row.get('Photocat')
        photocat_str = str(photocat) if pd.notna(photocat) else "N/A"
        hover_text = (
            f"<b>Index: {idx} | Split: {split}</b><br>────────────────<br>"
            f"<b> pk value:</b> {k_str}<br>"
            f"<b> Photocatalyst:</b> {photocat_str}<br>────────────────<br>"
            f"<b> Experimental Conditions:</b><br>"
        )
        for exp_col in numerical_features:
            if exp_col in row.index and pd.notna(row.get(exp_col)):
                exp_lower = exp_col.lower()
                if 'dosage' in exp_lower:
                    display_name = 'Dosage (g/L)'
                elif 'concentration' in exp_lower:
                    display_name = 'Concentration (ppm)'
                elif 'ph' in exp_lower:
                    display_name = 'pH'
                elif 'light' in exp_lower:
                    display_name = 'Light Intensity'
                else:
                    display_name = exp_col
                val = row.get(exp_col)
                val_str = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                hover_text += f"   • {display_name}: {val_str}<br>"
        hover_texts.append(hover_text)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n    Running t-SNE 3D (perplexity=30, n_iter=1000)...")
    tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto', n_iter=1000, random_state=42, init='pca')
    X_tsne = tsne.fit_transform(X_scaled)
    fig_tsne_3d = go.Figure()
    fig_tsne_3d.add_trace(go.Scatter3d(
        x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2], mode='markers',
        marker=dict(size=6, color=targets, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='pk', x=1.02, titleside='right', tickformat='.4f'),
                    line=dict(width=0.5, color='black')),
        text=hover_texts, hoverinfo='text'
    ))
    clean_name = feature_type.lower().replace(' ', '_')
    fig_tsne_3d.update_layout(
        title=dict(text=f'<b>t-SNE 3D — {feature_type}</b><br><sup>perplexity=30, n_iter=1000</sup>',
                  font=dict(size=14), x=0.5),
        scene=dict(xaxis=dict(title='Components_1'), yaxis=dict(title='Components_2'), zaxis=dict(title='Components_3'), bgcolor='white'),
        width=1000, height=800, hoverlabel=dict(bgcolor='white', font_size=11), margin=dict(l=50, r=50, t=100, b=50)
    )
    fig_tsne_3d.write_html(f'tsne_3d_{clean_name}_k_color.html')
    print(f"       Saved: tsne_3d_{clean_name}_k_color.html")

    print("\n    Running t-SNE 2D (perplexity=30, n_iter=1000)...")
    tsne2 = TSNE(n_components=2, perplexity=30, learning_rate='auto', n_iter=1000, random_state=42, init='pca')
    X_tsne2 = tsne2.fit_transform(X_scaled)
    fig_tsne_2d = go.Figure()
    fig_tsne_2d.add_trace(go.Scatter(
        x=X_tsne2[:, 0], y=X_tsne2[:, 1], mode='markers',
        marker=dict(size=8, color=targets, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='pk', x=1.02, titleside='right', tickformat='.4f'),
                    line=dict(width=0.5, color='black')),
        text=hover_texts, hoverinfo='text'
    ))
    fig_tsne_2d.update_layout(
        title=dict(text=f'<b>t-SNE 2D — {feature_type}</b><br><sup>perplexity=30, n_iter=1000</sup>',
                  font=dict(size=14), x=0.5),
        xaxis=dict(title='t-SNE1', gridcolor='lightgray'), yaxis=dict(title='t-SNE2', gridcolor='lightgray'),
        plot_bgcolor='white', width=900, height=700, hoverlabel=dict(bgcolor='white', font_size=11),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    fig_tsne_2d.write_html(f'tsne_2d_{clean_name}_k_color.html')
    print(f"       Saved: tsne_2d_{clean_name}_k_color.html")

