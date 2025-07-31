# JCTA GraphGym - Custom Graph Neural Network Framework

This repository contains a customized version of PyTorch Geometric's GraphGym framework with additional GNN models and datasets for node classification tasks.

## 🚀 Quick Start

### Custom GraphGym Models

This project extends GraphGym with three popular Graph Neural Network models:

#### 1. Graph Convolutional Network (GCN)
- **Paper**: [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907) by Kipf & Welling (2017)
- **Best for**: Citation networks (Cora, CiteSeer, PubMed)

#### 2. GraphSAGE
- **Paper**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216) by Hamilton et al. (2017)
- **Best for**: Large-scale graphs, inductive learning tasks

#### 3. Graph Attention Network (GAT)
- **Paper**: [Graph Attention Networks](https://arxiv.org/pdf/1710.10903) by Veličković et al. (2018)
- **Best for**: Tasks requiring adaptive neighbor importance

### Running Experiments

The project includes three custom GNN models for node classification:

- **GCN** (Graph Convolutional Network) - Kipf & Welling (2017)
- **GraphSAGE** (Graph SAmple and aggreGatE) - Hamilton et al. (2017)
- **GAT** (Graph Attention Network) - Veličković et al. (2018)

#### Quick Experiment

To run a GraphSAGE experiment with the Elliptic dataset:

```bash
cd pytorch_geometric/graphgym
python main.py --cfg configs/pyg/graphsage_node.yaml
```

#### Available Configurations

| Model | Configuration File | Description |
|-------|-------------------|-------------|
| **GCN** | `configs/pyg/gcn_node.yaml` | Graph Convolutional Network |
| **GraphSAGE** | `configs/pyg/graphsage_node.yaml` | GraphSAGE (Inductive Learning) |
| **GAT** | `configs/pyg/gat_node.yaml` | Graph Attention Network |

#### Running All Models

```bash
# Navigate to GraphGym directory
cd pytorch_geometric/graphgym

# Run all three models
python main.py --cfg configs/pyg/gcn_node.yaml
python main.py --cfg configs/pyg/graphsage_node.yaml
python main.py --cfg configs/pyg/gat_node.yaml
```

#### Customizing Experiments

**Change Dataset**: Edit the configuration file and modify:
```yaml
dataset:
  name: Elliptic  # Options: Elliptic, Reddit, Yelp, Cora, CiteSeer, PubMed
```

**Adjust Model Parameters**: Modify the GNN section:
```yaml
gnn:
  dim_inner: 64         # Hidden dimension
  layers_mp: 2          # Number of layers
  dropout: 0.5          # Dropout rate
```

**Training Settings**: Adjust optimization parameters:
```yaml
optim:
  base_lr: 0.01         # Learning rate
  max_epoch: 200        # Training epochs
  weight_decay: 5e-4    # Weight decay
```

#### Results and Analysis

Results are saved in `pytorch_geometric/graphgym/results/` with the following structure:

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

**Key Result Files**:
- `agg/train/best.json`: Best training metrics
- `agg/val/best.json`: Best validation metrics
- `config.yaml`: Configuration used for the experiment

#### Experimental Results

On the Elliptic dataset (Bitcoin transaction fraud detection):

| Model | Validation Accuracy | Precision | Recall | F1-Score | AUC | Status |
|-------|-------------------|-----------|--------|----------|-----|--------|
| **GraphSAGE** | **97.00%** | **96.74%** | **66.40%** | **78.74%** | **93.16%** | ✅ Working |
| **GCN** | 94.38% | 90.86% | 36.46% | 52.03% | 86.94% | ✅ Working |
| **GAT** | - | - | - | - | - | ❌ Memory Error |

**Key Findings:**
- **GraphSAGE** achieved the best overall performance with balanced precision and recall
- **GCN** showed good accuracy but lower recall for illicit transaction detection
- **GAT** encountered memory issues and needs optimization for this dataset size

*For detailed results, see [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)*

#### Troubleshooting

**Common Issues**:
1. **Import Errors**: Ensure virtual environment is activated
2. **CUDA Issues**: Models automatically use CPU if CUDA unavailable
3. **Memory Issues**: Reduce `batch_size` or `dim_inner`
4. **Configuration Errors**: Check YAML syntax and parameter names

**Performance Tips**:
1. **GPU Usage**: Models automatically use CUDA if available
2. **Batch Size**: Adjust based on GPU memory
3. **Learning Rate**: Start with 0.01 and adjust based on convergence
4. **Early Stopping**: Monitor validation metrics to prevent overfitting

### Supported Datasets

All models support the following datasets:
- **Citation Networks**: Cora, CiteSeer, PubMed
- **Social Networks**: Reddit, Yelp
- **Financial Networks**: Elliptic

## 📁 Project Structure

```
jcta-graphgym/
├── pytorch_geometric/
│   ├── graphgym/
│   │   ├── custom_graphgym/          # Custom implementations
│   │   │   ├── network/              # GNN model implementations
│   │   │   │   ├── gcn.py           # GCN model
│   │   │   │   ├── graphsage.py     # GraphSAGE model
│   │   │   │   └── gat.py           # GAT model
│   │   │   ├── config/              # Model configurations
│   │   │   ├── loader/              # Dataset loaders
│   │   │   └── README.md            # Custom GraphGym documentation
│   │   ├── configs/pyg/             # YAML configuration files
│   │   └── main.py                  # Main training script
│   └── README.md                    # Original PyTorch Geometric README
└── README.md                        # This file
```

## 📊 Results

Results will be saved in the `results/` directory with the following structure:
```
results/
├── gcn_node/
│   ├── logs/
│   ├── checkpoints/
│   └── results.json
├── graphsage_node/
└── gat_node/
```

## 🔧 Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+

### Setup

#### Option 1: Automated Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/IoTDataAtelier/jcta-graphgym.git
cd jcta-graphgym

# Run the installation script
./install.sh
```

#### Option 2: Manual Installation
```bash
# Clone the repository
git clone https://github.com/IoTDataAtelier/jcta-graphgym.git
cd jcta-graphgym

# Install PyTorch
pip install torch>=2.0.0

# Install local torch_geometric (this project uses a local version)
pip install -e ./pytorch_geometric

# Install other dependencies
pip install -r requirements.txt

# Optional: Install PyTorch Geometric extensions for better performance
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
```

### Local torch_geometric Usage
This project uses a local version of PyTorch Geometric located in the `./pytorch_geometric` directory. This allows you to:
- Use the exact version that was tested with the custom models
- Make modifications to PyTorch Geometric if needed
- Ensure compatibility with the custom GraphGym implementations

## 📚 Documentation

For detailed information about the custom GraphGym implementations, see:
- [Custom GraphGym README](pytorch_geometric/graphgym/custom_graphgym/README.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - The original framework
- [GraphGym](https://github.com/snap-stanford/graphgym) - The original GraphGym implementation

---

<details>
<summary><b>📖 Original PyTorch Geometric Documentation</b></summary>

# PyTorch Geometric (PyG)

**PyG** *(PyTorch Geometric)* is a library built upon [PyTorch](https://pytorch.org/) to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.

## Library Highlights

- **Easy-to-use and unified API**: All it takes is 10-20 lines of code to get started with training a GNN model
- **Comprehensive and well-maintained GNN models**: Most of the state-of-the-art Graph Neural Network architectures have been implemented
- **Great flexibility**: Existing PyG models can easily be extended for conducting your own research with GNNs
- **Large-scale real-world GNN models**: Support for learning on diverse types of graphs

## Quick Tour

### Train your own GNN model

```python
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='.', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, 16, dataset.num_classes)
```

### Create your own GNN layer

```python
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")
        self.mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i):
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)
```

## Installation

### Basic Installation

```bash
pip install torch_geometric
```

### Additional Libraries (Optional)

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
```

## Implemented GNN Models

### GNN Layers
- **[GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html)** - Graph Convolutional Networks
- **[GATConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html)** - Graph Attention Networks
- **[SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html)** - GraphSAGE
- **[GINConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html)** - Graph Isomorphism Networks
- And many more...

### Pooling Layers
- **[Top-K Pooling](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.TopKPooling.html)**
- **[DiffPool](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.dense_diff_pool.html)**
- And many more...

### Scalable GNNs
- **[NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader)**
- **[ClusterGCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.ClusterLoader)**
- **[GraphSAINT](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.GraphSAINTSampler)**

## Documentation

- **[Documentation](https://pytorch-geometric.readthedocs.io)**
- **[Paper](https://arxiv.org/abs/1903.02428)**
- **[Colab Notebooks and Video Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)**

## Cite

```bibtex
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```

For more information, visit the [original PyTorch Geometric repository](https://github.com/pyg-team/pytorch_geometric).

</details> 