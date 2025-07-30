from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('gcn')
def set_cfg_gcn(cfg):
    r"""Configuration for Graph Convolutional Network (GCN)
    Based on: Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks.
    """
    # ----------------------------------------------------------------------- #
    # GCN specific options
    # ----------------------------------------------------------------------- #
    
    # Model architecture
    cfg.gnn.layer_type = 'gcn'
    cfg.gnn.num_layers = 2
    cfg.gnn.hidden_dim = 64
    cfg.gnn.dropout = 0.5
    
    # Training parameters
    cfg.optim.base_lr = 0.01
    cfg.optim.weight_decay = 5e-4
    cfg.optim.max_epoch = 200
    
    # Model specific
    cfg.model.loss_fun = 'cross_entropy' 