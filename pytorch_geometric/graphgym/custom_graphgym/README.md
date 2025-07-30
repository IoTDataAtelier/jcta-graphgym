# Custom GraphGym GNN Models

This directory contains custom implementations of three popular Graph Neural Network models for node classification tasks in GraphGym.

## Models Implemented

### 1. Graph Convolutional Network (GCN)
- **Paper**: [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907) by Kipf & Welling (2017)
- **Key Features**: 
  - Spectral-based approach using graph Laplacian
  - Efficient message passing with normalized adjacency matrix
  - Good for transductive learning tasks
- **Best for**: Citation networks (Cora, CiteSeer, PubMed)

### 2. GraphSAGE
- **Paper**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216) by Hamilton et al. (2017)
- **Key Features**:
  - Inductive learning approach
  - Neighborhood sampling for scalability
  - Multiple aggregation functions (mean, max, LSTM)
- **Best for**: Large-scale graphs, inductive learning tasks

### 3. Graph Attention Network (GAT)
- **Paper**: [Graph Attention Networks](https://arxiv.org/pdf/1710.10903) by Veličković et al. (2018)
- **Key Features**:
  - Attention mechanism for weighted neighbor aggregation
  - Multi-head attention for stability
  - Learnable attention weights
- **Best for**: Tasks requiring adaptive neighbor importance

## Usage

### Running Experiments

1. **GCN on Cora**:
```bash
cd pytorch_geometric/graphgym
python main.py --cfg configs/pyg/gcn_node.yaml
```

2. **GraphSAGE on Reddit**:
```bash
python main.py --cfg configs/pyg/graphsage_node.yaml
```

3. **GAT on Cora**:
```bash
python main.py --cfg configs/pyg/gat_node.yaml
```

### Configuration Files

Each model has its own configuration file:
- `configs/pyg/gcn_node.yaml` - GCN configuration
- `configs/pyg/graphsage_node.yaml` - GraphSAGE configuration  
- `configs/pyg/gat_node.yaml` - GAT configuration

### Supported Datasets

All models support the following datasets:
- **Citation Networks**: Cora, CiteSeer, PubMed
- **Social Networks**: Reddit, Yelp
- **Financial Networks**: Elliptic

### Model Parameters

#### GCN
- `hidden_dim`: Hidden layer dimension (default: 64)
- `num_layers`: Number of GCN layers (default: 2)
- `dropout`: Dropout rate (default: 0.5)

#### GraphSAGE
- `hidden_dim`: Hidden layer dimension (default: 64)
- `num_layers`: Number of GraphSAGE layers (default: 2)
- `dropout`: Dropout rate (default: 0.5)
- `aggregator`: Aggregation function (default: 'mean', options: 'mean', 'max', 'lstm')

#### GAT
- `hidden_dim`: Hidden layer dimension (default: 64)
- `num_layers`: Number of GAT layers (default: 2)
- `dropout`: Dropout rate (default: 0.6)
- `heads`: Number of attention heads (default: 8)
- `concat`: Whether to concatenate attention heads (default: true)
- `negative_slope`: LeakyReLU negative slope (default: 0.2)

## Implementation Details

### Architecture
Each model follows the standard GraphGym architecture:
1. **Input Layer**: Processes node features
2. **GNN Layers**: Stack of graph convolution/attention layers
3. **Output Head**: Task-specific prediction layer

### Activation Functions
- **GCN & GraphSAGE**: ReLU activation between layers
- **GAT**: ELU activation between layers (as per original paper)

### Training
- **Optimizer**: Adam
- **Learning Rate**: Model-specific (GCN: 0.01, GAT: 0.005, GraphSAGE: 0.01)
- **Weight Decay**: 5e-4 for all models
- **Scheduler**: ReduceLROnPlateau with patience=10

## Customization

To modify model parameters, edit the corresponding YAML configuration file or use command-line options:

```bash
python main.py --cfg configs/pyg/gcn_node.yaml opts gnn.hidden_dim 128 gnn.num_layers 3
```

## Results

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

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. NIPS.
3. Veličković, P., et al. (2018). Graph attention networks. ICLR. 