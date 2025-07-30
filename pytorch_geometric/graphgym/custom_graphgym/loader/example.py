from torch_geometric.datasets import QM7b, EllipticBitcoinDataset, Reddit, Yelp
from torch_geometric.graphgym.register import register_loader

@register_loader('example')
def load_dataset_example(format, name, dataset_dir):
    dataset_dir = f'{dataset_dir}/{name}'
    if format == 'PyG':
        if name == 'QM7b':
            dataset_raw = QM7b(dataset_dir)
            return dataset_raw
        if name=='Elliptic':
            dataset_raw = EllipticBitcoinDataset(dataset_dir)
            return dataset_raw
        if name == 'Reddit':
            dataset_raw = Reddit(dataset_dir)
            return dataset_raw
        if name == 'Yelp':
            dataset_raw = Yelp(dataset_dir)
            return dataset_raw