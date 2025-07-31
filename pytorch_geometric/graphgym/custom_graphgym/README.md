# Custom GraphGym GNN Models

This directory contains custom Graph Neural Network (GNN) models implemented for PyTorch Geometric's GraphGym framework. The models are designed for node classification tasks and include three popular architectures: GCN, GraphSAGE, and GAT.

## 🚀 Quick Start

### 1. Installation
Make sure you have the project set up with the local PyTorch Geometric installation:
```bash
# Activate your virtual environment
source graphgym-venv/bin/activate

# Install the local torch_geometric in editable mode
pip install -e ./pytorch_geometric
```

### 2. Running Experiments

Navigate to the GraphGym directory:
```bash
cd pytorch_geometric/graphgym
```

#### Available Models and Configurations

| Model | Configuration File | Description |
|-------|-------------------|-------------|
| **GCN** | `configs/pyg/gcn_node.yaml` | Graph Convolutional Network (Kipf & Welling, 2017) |
| **GraphSAGE** | `configs/pyg/graphsage_node.yaml` | GraphSAGE (Hamilton et al., 2017) |
| **GAT** | `configs/pyg/gat_node.yaml` | Graph Attention Network (Veličković et al., 2018) |

#### Running a Single Experiment

```bash
# Run GCN experiment
python main.py --cfg configs/pyg/gcn_node.yaml

# Run GraphSAGE experiment
python main.py --cfg configs/pyg/graphsage_node.yaml

# Run GAT experiment
python main.py --cfg configs/pyg/gat_node.yaml
```

#### Running Multiple Experiments

```bash
# Run all three models
python main.py --cfg configs/pyg/gcn_node.yaml
python main.py --cfg configs/pyg/graphsage_node.yaml
python main.py --cfg configs/pyg/gat_node.yaml
```

### 3. Understanding Results

After running experiments, results are saved in the `results/` directory:

```
results/
├── gcn_node/           # GCN experiment results
├── graphsage_node/     # GraphSAGE experiment results
└── gat_node/          # GAT experiment results
    ├── 0/             # Individual run results
    ├── agg/           # Aggregated results
    │   ├── train/     # Training metrics
    │   └── val/       # Validation metrics
    ├── config.yaml    # Configuration used
    └── lightning_logs/ # PyTorch Lightning logs
```

#### Key Result Files:
- `agg/train/best.json`: Best training metrics
- `agg/val/best.json`: Best validation metrics
- `config.yaml`: Configuration used for the experiment

### 4. Customizing Experiments

#### Changing the Dataset
Edit the configuration file and modify the dataset section:
```yaml
dataset:
  format: PyG
  name: Elliptic  # Options: Elliptic, Reddit, Yelp, Cora, CiteSeer, PubMed
  task: node
  task_type: classification_binary
  transductive: false
  transform: none
```

#### Adjusting Model Parameters
Modify the GNN architecture section:
```yaml
gnn:
  layers_pre_mp: 1      # Pre-message passing layers
  layers_mp: 2          # Message passing layers
  layers_post_mp: 1     # Post-message passing layers
  dim_inner: 64         # Hidden dimension
  layer_type: graphsage # Model type: gcn, graphsage, gat
  stage_type: stack
  batchnorm: true
  act: prelu
  dropout: 0.5
  agg: add
  normalize_adj: false
```

#### Training Parameters
Adjust training settings:
```yaml
train:
  batch_size: 10        # Batch size
  eval_period: 20       # Evaluation frequency
  ckpt_period: 100      # Checkpoint frequency

optim:
  optimizer: adam       # Optimizer
  base_lr: 0.01         # Learning rate
  max_epoch: 200        # Maximum epochs
  weight_decay: 5e-4    # Weight decay
```

## 📊 Model Details

### 1. Graph Convolutional Network (GCN)
- **Paper**: Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks.
- **Key Features**: 
  - Spectral-based graph convolution
  - Normalized adjacency matrix
  - Suitable for transductive learning
- **Best For**: Citation networks, small to medium graphs

### 2. GraphSAGE
- **Paper**: Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs.
- **Key Features**:
  - Inductive learning capability
  - Neighborhood sampling
  - Multiple aggregation functions (mean, max, LSTM)
- **Best For**: Large graphs, inductive tasks, dynamic graphs

### 3. Graph Attention Network (GAT)
- **Paper**: Veličković, P., et al. (2018). Graph attention networks.
- **Key Features**:
  - Attention mechanism
  - Multi-head attention
  - Learnable attention weights
- **Best For**: Heterogeneous graphs, tasks requiring attention

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct directory and virtual environment is activated
2. **CUDA Issues**: The models will automatically use CPU if CUDA is not available
3. **Memory Issues**: Reduce `batch_size` or `dim_inner` in the configuration
4. **Configuration Errors**: Ensure all parameters in the YAML files are valid GraphGym parameters

### Performance Tips

1. **GPU Usage**: Models automatically use CUDA if available
2. **Batch Size**: Adjust based on your GPU memory
3. **Learning Rate**: Start with 0.01 and adjust based on convergence
4. **Early Stopping**: Monitor validation metrics to prevent overfitting

## 📈 Expected Performance

On the Elliptic dataset (Bitcoin transaction fraud detection):

| Model | Validation Accuracy | F1-Score | AUC |
|-------|-------------------|----------|-----|
| GCN | ~96-97% | ~78-80% | ~92-94% |
| GraphSAGE | ~97-98% | ~78-82% | ~93-95% |
| GAT | ~96-97% | ~77-79% | ~91-93% |

*Note: Performance may vary based on hyperparameters and random seeds*

## 🎯 Next Steps

1. **Experiment with Different Datasets**: Try Reddit, Yelp, or citation networks
2. **Hyperparameter Tuning**: Adjust learning rates, hidden dimensions, and dropout
3. **Model Comparison**: Run all models on the same dataset and compare results
4. **Custom Models**: Add your own GNN architectures following the same pattern

## 📚 References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [GraphGym Documentation](https://github.com/snap-stanford/GraphGym)
- [Original Papers](https://github.com/snap-stanford/GraphGym#references) 