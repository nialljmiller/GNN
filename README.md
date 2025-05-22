# PRIMVS Variable Star Classification

A computational framework for multi-class classification using machine learning on time-series survey data. Implements XGBoost and Graph Neural Networks with contrastive learning embeddings for large-scale astronomical datasets.

## Method Comparison

| Approach | Computational Complexity | Memory Usage | GPU Scaling | Best Use Case |
|----------|--------------------------|--------------|-------------|---------------|
| XGBoost | O(n log n) per tree | Moderate | Multi-GPU | Tabular data, interpretability |
| GNN | O(E + V) per layer | High | Single-GPU | Graph structure, embeddings |

## Core Components

### Cluster job submission for requesting computational resources:

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
---

### XGBoost implementation with advanced preprocessing:

```python
# Basic usage
python GXGB.py train_data.fits test_data.fits predictions.csv
```

**Key computational features:**
- Quantile clipping, robust scaling, median imputation
- Square-root weighted sampling for imbalanced data
- Auto-detection and multi-GPU utilisation
- Early stopping with validation monitoring

---

### GCN architecture with KNN graph construction:

```python
# Train GNN classifier
python GNN.py train_data.fits test_data.fits gnn_predictions.csv
```

**Implementation details:**
- FAISS-accelerated KNN (k=8 default) -- this is stupid
- 2-layer GCN with dropout regularisation
- Class-weighted loss, adaptive learning rate
- Memory-efficient inference for large graphs

---

### vis.py

```python
from vis import *

# Generate evaluation dashboard
plot_bailey_diagram(results_df, "predicted_class")
plot_galactic_distribution(results_df, "predicted_class") 
plot_confidence_entropy(results_df, "predicted_class")
```


---

### Input format
FITS files with training labels in `best_class_name` column

## Computational Workflow

```bash
./iGPU.sh medium

python GXGB.py PRIMVS_train.fits PRIMVS_test.fits xgb_results.csv

python GNN.py PRIMVS_train.fits PRIMVS_test.fits gnn_results.csv
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

vis.py automatically generates evaluation plots in `figures/` directory.

