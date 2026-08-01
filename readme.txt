# Photocatalytic Degradation Rate Prediction using Multi-Modal Graph Neural Networks

## Overview

This project presents a multi-modal Graph Neural Network (GNN) framework for predicting photocatalytic degradation rate constants (k) from:

- Organic pollutant molecular structures (SMILES)
- Photocatalyst crystal structures (CIF files)
- Experimental conditions

The model integrates molecular graphs, crystal graphs, global descriptors, and experimental parameters into a unified architecture based on Graph Attention Networks (GAT).

---

## Features

### Molecular Representation
- Molecular graph construction from SMILES
- RDKit atom features
- Bond features including:
  - Bond type
  - Bond stereochemistry
  - Conjugation
  - Bond length
- 25 global molecular descriptors

### Crystal Representation
- Crystal graph generation from CIF files
- CGCNN-based crystal featurization
- Crystal global descriptors:
  - Lattice parameters
  - Packing density
  - Bond lengths
  - Bond angles
  - Coordination number
  - Wyckoff positions
  - Electronegativity differences
  - Crystal symmetry information
  - Band gap information

### Experimental Conditions
- Catalyst dosage
- Pollutant concentration
- pH
- Light source encoding

### Deep Learning Architecture
- Molecular GAT branch
- Crystal GAT branch
- Experimental feature branch
- Feature interaction modules
- Multi-modal fusion network
- Regression output for degradation rate constant prediction

---

## Project Structure

```text
project/
│
├── main.py
├── data_processing.py
├── feature_engineering.py
├── model.py
├── cif_files/
├── data/
└── README.md
```

### Files Description

#### main.py
Main execution script:
- Data loading
- Dataset splitting
- Model training
- Validation
- Testing
- Evaluation

#### data_processing.py
Data handling utilities:
- Excel loading
- Column detection
- Dataset creation
- Data normalization
- DataLoader preparation

#### feature_engineering.py
Feature extraction:
- Molecular graph generation
- Crystal graph generation
- Global descriptor calculation
- CIF processing

#### model.py
Model architecture:
- Graph Attention Networks
- Fusion layers
- Prediction head
- Evaluation utilities

---

## Requirements

### Python

Python 3.10 or later is recommended.

### Dependencies

```bash
pip install torch
pip install torch-geometric
pip install rdkit
pip install deepchem
pip install pymatgen
pip install pandas
pip install numpy
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install plotly
pip install umap-learn
pip install openpyxl
```

Or:

```bash
pip install -r requirements.txt
```

---

## Dataset Format

The Excel file should contain:

| Column | Description |
|----------|------------|
| SMILES | Pollutant molecular structure |
| Photocat | Photocatalyst name |
| k | Experimental degradation rate constant |
| Dosage | Catalyst dosage |
| Concentration | Initial pollutant concentration |
| pH | Solution pH |
| Light | Light source category |

The program automatically attempts to detect columns.

---

## Crystal Structure Files

All photocatalyst crystal structures must be stored as CIF files.

Example:

```text
cif_files/
├── TiO2.cif
├── ZnO.cif
├── WO3.cif
├── Fe2O3.cif
└── SnO2.cif
```

The CIF filename should match the photocatalyst name used in the dataset.

---

## Model Architecture

The proposed framework consists of:

1. Molecular Graph Encoder
   - Three GAT layers
   - GraphNorm
   - Global Mean Pooling

2. Crystal Graph Encoder
   - Three GAT layers
   - GraphNorm
   - Global Mean Pooling

3. Global Feature Encoders
   - Molecular descriptors
   - Crystal descriptors

4. Experimental Feature Encoder

5. Feature Interaction Module
   - Molecule × Experiment
   - Crystal × Experiment
   - Optional Molecule × Crystal interaction

6. Fusion Network
   - Fully connected layers
   - Dropout regularization
   - Regression output

---

## Training Strategy

- Random seed fixing for reproducibility
- Train/Validation/Test split
- Standardization using training data only
- Learning rate scheduling
- Early stopping support (optional)
- Regression loss optimization

---

## Evaluation Metrics

The model is evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Output

The model generates:

- Predicted degradation rate constants
- Training and validation metrics
- Calculated vs Experimental plots
- Regression statistics

Example:

```text
R² = 0.92
RMSE = 0.08
MAE = 0.05
```

---

## Reproducibility

Random seeds are fixed for:

- Python
- NumPy
- PyTorch

to ensure reproducible results.

---

## Citation

If you use this code in academic work, please cite the corresponding publication or repository.

---

## Author

Developed for photocatalytic degradation modeling using multi-modal graph neural networks integrating molecular, crystal, and experimental information.