import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import GATConv


@register_network('gat')
class GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) implementation based on:
    Veličković, P., Casanova, A., Liò, P., Cucurull, G., Romero, A., & Bengio, Y. (2018).
    Graph attention networks. International Conference on Learning Representations (ICLR).
    """
    def __init__(self, dim_in, dim_out, num_layers=2, hidden_dim=None, dropout=0.6, 
                 heads=8, concat=True, negative_slope=0.2):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = dim_in
            
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        
        # Calculate output dimensions for each layer
        if concat:
            layer_out_dim = hidden_dim // heads
        else:
            layer_out_dim = hidden_dim
            
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer: input dimension to hidden dimension
        self.convs.append(GATConv(dim_in, layer_out_dim, heads=heads, 
                                 dropout=dropout, concat=concat, 
                                 negative_slope=negative_slope))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, layer_out_dim, heads=heads,
                                     dropout=dropout, concat=concat,
                                     negative_slope=negative_slope))
            
        # Output layer (if more than 1 layer)
        if num_layers > 1:
            # Final layer: single attention head for output
            self.convs.append(GATConv(hidden_dim, layer_out_dim, heads=1,
                                     dropout=dropout, concat=False,
                                     negative_slope=negative_slope))
        
        # Task-specific head
        GNNHead = register.head_dict[cfg.dataset.task]
        self.post_mp = GNNHead(dim_in=layer_out_dim, dim_out=dim_out)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        # Apply GAT layers with ELU activation and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation after last conv
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        batch.x = x
        batch = self.post_mp(batch)
        
        return batch 