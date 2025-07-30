from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('gat')
def set_cfg_gat(cfg):
    r"""Configuration for Graph Attention Network (GAT)
    Based on: Veličković, P., Casanova, A., Liò, P., Cucurull, G., Romero, A., & Bengio, Y. (2018).
    Graph attention networks.
    """
    # ----------------------------------------------------------------------- #
    # GAT specific options
    # ----------------------------------------------------------------------- #
    
    # Model architecture
    cfg.gnn.layer_type = 'gat'
    cfg.gnn.num_layers = 2
    cfg.gnn.hidden_dim = 64
    cfg.gnn.dropout = 0.6
    cfg.gnn.heads = 8
    cfg.gnn.concat = True
    cfg.gnn.negative_slope = 0.2
    
    # Training parameters
    cfg.optim.base_lr = 0.005
    cfg.optim.weight_decay = 5e-4
    cfg.optim.max_epoch = 200
    
    # Model specific
    cfg.model.loss_fun = 'cross_entropy' 