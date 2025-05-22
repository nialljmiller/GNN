# PRIMVS Variable Star Classification

A computational framework for multi-class classification using machine learning on time-series survey data. Implements XGBoost and Graph Neural Networks with contrastive learning embeddings for large-scale astronomical datasets.

## Core Objectives
- **Multi-method classification**: Gradient boosting and graph-based approaches
- **High-dimensional feature engineering**: 128-dimensional contrastive embeddings
- **Scalable computing**: GPU cluster resource management
- **Performance visualisation**: Comprehensive model evaluation tools

## Method Comparison

| Approach | Computational Complexity | Memory Usage | GPU Scaling | Best Use Case |
|----------|--------------------------|--------------|-------------|---------------|
| XGBoost | O(n log n) per tree | Moderate | Multi-GPU | Tabular data, interpretability |
| GNN | O(E + V) per layer | High | Single-GPU | Graph structure, embeddings |

## Core Components

### 1. Resource Management (`iGPU.sh`)
Cluster job submission for requesting computational resources:

```bash
# Request computational tiers
./iGPU.sh low     # A30 GPU, 64GB RAM
./iGPU.sh medium  # L40S GPU, 128GB RAM  
./iGPU.sh high    # H100 GPU, 256GB RAM
```

**Typical workflow:**
```bash
./iGPU.sh medium
python GXGB.py train.fits test.fits output.csv
```

### 2. Gradient Boosting (`GXGB.py`)
XGBoost implementation with advanced preprocessing:

```python
# Basic usage
python GXGB.py train_data.fits test_data.fits predictions.csv
```

**Key computational features:**
- **Preprocessing pipeline**: Quantile clipping, robust scaling, median imputation
- **Class balancing**: Square-root weighted sampling for imbalanced data
- **GPU acceleration**: Auto-detection and multi-GPU utilisation
- **Regularisation**: Early stopping with validation monitoring

### 3. Graph Neural Networks (`GNN.py`)
GCN architecture with KNN graph construction:

```python
# Train GNN classifier
python GNN.py train_data.fits test_data.fits gnn_predictions.csv
```

**Implementation details:**
- **Graph construction**: FAISS-accelerated KNN (k=8 default)
- **Architecture**: 2-layer GCN with dropout regularisation
- **Training**: Class-weighted loss, adaptive learning rate
- **Batching**: Memory-efficient inference for large graphs

### 4. Visualisation Suite (`vis.py`)
Comprehensive plotting and analysis functions:

```python
from vis import *

# Generate evaluation dashboard
plot_bailey_diagram(results_df, "predicted_class")
plot_galactic_distribution(results_df, "predicted_class") 
plot_confidence_entropy(results_df, "predicted_class")
```

**Available analysis tools:**
- **Performance metrics**: Confusion matrices, classification reports
- **Feature analysis**: Importance plots, correlation heatmaps
- **Model interpretation**: Confidence distributions, embedding visualisations

## Data Requirements

### Feature Engineering Pipeline
```python
# Automatically detected feature sets:
variability_metrics = ["MAD", "eta_e", "true_amplitude", "stet_k", ...]
colour_indices = ["j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag", ...]
temporal_features = ["true_period", "ls_fap", "pdm_fap", ...]
contrastive_embeddings = ["0", "1", "2", ..., "127"]  # 128D learned features
spatial_coords = ["l", "b", "parallax", "pmra", "pmdec"]
```

### Input format
FITS files with training labels in `best_class_name` column

## Computational Workflow

```bash
# 1. Allocate resources
./iGPU.sh medium

# 2. Run gradient boosting
python GXGB.py PRIMVS_train.fits PRIMVS_test.fits xgb_results.csv

# 3. Run graph neural network
python GNN.py PRIMVS_train.fits PRIMVS_test.fits gnn_results.csv

# 4. Analyse results
ls figures/  # Evaluation plots generated automatically
```

## Performance Optimisation

### Computational scaling
- **Memory management**: Batched processing for large datasets
- **GPU utilisation**: Automatic detection and efficient allocation
- **Preprocessing**: Robust outlier handling and feature standardisation

### Model configuration
```python
# XGBoost parameters (auto-configured)
xgb_params = {
    'learning_rate': 0.01,
    'max_depth': 10,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

# GNN architecture (modifiable in script)
gnn_config = {
    'hidden_dim': 128,
    'dropout': 0.5,
    'k_neighbors': 8,
    'batch_size': 4096
}
```

## Output interpretation

Both classifiers generate structured results:
```python
output_columns = [
    'predicted_class',      # Classification result
    'confidence',           # Model certainty [0,1]
    'predicted_class_id',   # Numeric class identifier
    # ... original input features preserved
]
```

Visualisation suite automatically generates evaluation plots in `figures/` directory.

---

**Technical note**: Framework requires 128-dimensional contrastive learning embeddings (columns "0"-"127") for optimal performance. These learned representations capture complex feature relationships through self-supervised training.
