from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('graphsage')
def set_cfg_graphsage(cfg):
    r"""Configuration for GraphSAGE
    Based on: Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs.
    """
    # ----------------------------------------------------------------------- #
    # GraphSAGE specific options
    # ----------------------------------------------------------------------- #
    
    # Model architecture
    cfg.gnn.layer_type = 'graphsage'
    cfg.gnn.num_layers = 2
    cfg.gnn.hidden_dim = 64
    cfg.gnn.dropout = 0.5
    cfg.gnn.aggregator = 'mean'  # Options: 'mean', 'max', 'lstm'
    
    # Training parameters
    cfg.optim.base_lr = 0.01
    cfg.optim.weight_decay = 5e-4
    cfg.optim.max_epoch = 200
    
    # Model specific
    cfg.model.loss_fun = 'cross_entropy' 