import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import SAGEConv


@register_network('graphsage')
class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE implementation based on:
    Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs.
    Neural Information Processing Systems (NIPS).
    """
    def __init__(self, dim_in, dim_out, num_layers=2, hidden_dim=None, dropout=0.5, 
                 aggregator='mean'):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = dim_in
            
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # First layer: input dimension to hidden dimension
        self.convs.append(SAGEConv(dim_in, hidden_dim, aggr=self.aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=self.aggregator))
            
        # Output layer (if more than 1 layer)
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=self.aggregator))
        
        # Task-specific head
        GNNHead = register.head_dict[cfg.dataset.task]
        self.post_mp = GNNHead(dim_in=hidden_dim, dim_out=dim_out)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        # Apply GraphSAGE layers with ReLU activation and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation after last conv
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        batch.x = x
        batch = self.post_mp(batch)
        
        return batch 