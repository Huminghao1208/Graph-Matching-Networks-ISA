import collections
#from dataset import GraphEditDistanceDataset, FixedGraphEditDistanceDataset
from graphembeddingnetwork import GraphEmbeddingNet, GraphEncoder, GraphAggregator
from graphmatchingnetwork import GraphMatchingNet
import copy
import torch
import random
import logging
import os
import numpy as np
import networkx as nx
import csv
from dataset import GraphEditDistanceDataset, FixedGraphEditDistanceDataset

##################################################################
#The following function and class are manually added: 

def read_adjacency_matrix(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        matrix = list(reader)
    return np.array(matrix, dtype=int)

def create_graph(adj_matrix):
    G=nx.DiGraph()
    num_nodes=adj_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j]!=0:
                G.add_edge(i,j)
    return G   

class CustomGraphEditDistanceDataset(GraphEditDistanceDataset):
    def __init__(self,input_folder,n_changes_positive,n_changes_negative):
        self.graphs=[]
        for file in os.listdir(input_folder):
            if file.endswith('.csv'):
                adj_matrix = read_adjacency_matrix(os.path.join(input_folder, file))
                graph = create_graph(adj_matrix)
                self.graphs.append(graph)
        
        self._k_pos = n_changes_positive
        self._k_neg = n_changes_negative
        self._permute = True
    def _get_graph(self):
        """Return a random graph from the loaded set."""
        return random.choice(self.graphs)

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])


def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def build_model(config, node_feature_dim, edge_feature_dim):
    """Create model for training and evaluation.

    Args:
      config: a dictionary of configs, like the one created by the
        `get_default_config` function.
      node_feature_dim: int, dimensionality of node features.
      edge_feature_dim: int, dimensionality of edge features.

    Returns:
      tensors: a (potentially nested) name => tensor dict.
      placeholders: a (potentially nested) name => tensor dict.
      AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

    Raises:
      ValueError: if the specified model or training settings are not supported.
    """
    config['encoder']['node_feature_dim'] = node_feature_dim
    config['encoder']['edge_feature_dim'] = edge_feature_dim

    encoder = GraphEncoder(**config['encoder'])
    aggregator = GraphAggregator(**config['aggregator'])
    if config['model_type'] == 'embedding':
        model = GraphEmbeddingNet(
            encoder, aggregator, **config['graph_embedding_net'])
    elif config['model_type'] == 'matching':
        model = GraphMatchingNet(
            encoder, aggregator, **config['graph_matching_net'])
    else:
        raise ValueError('Unknown model type: %s' % config['model_type'])

    optimizer = torch.optim.Adam((model.parameters()),
                                 lr=config['training']['learning_rate'], weight_decay=1e-5)

    return model, optimizer

def build_datasets(config):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)

    if config['data']['problem'] == 'graph_edit_distance':
        dataset_params = config['data']['dataset_params']
        logging.info(f"The dataset parameters are: {dataset_params}")
        validation_dataset_size = dataset_params['validation_dataset_size']
        del dataset_params['validation_dataset_size']
        training_set = GraphEditDistanceDataset(**dataset_params)
        logging.info(f"The training dataset info is: {training_set}")
        dataset_params['dataset_size'] = validation_dataset_size
        validation_set = FixedGraphEditDistanceDataset(**dataset_params)
    else:
        raise ValueError('Unknown problem type: %s' % config['data']['problem'])
    return training_set, validation_set


def build_datasets_from_CSV(config):
    """Build the training and evaluation datasets from CSV files"""
    config = copy.deepcopy(config)

    if config['data']['problem'] == 'graph_edit_distance':
        dataset_params = config['data']['dataset_params']
        input_folder = config['input_csv_folder']
        validation_dataset_size = dataset_params['validation_dataset_size']
        
        training_set = CustomGraphEditDistanceDataset(
            input_folder=input_folder,
            n_changes_positive=dataset_params['n_changes_positive'],
            n_changes_negative=dataset_params['n_changes_negative']
        )
        
        validation_set = FixedGraphEditDistanceDataset(
            n_nodes_range=dataset_params['n_nodes_range'],
            p_edge_range=dataset_params['p_edge_range'],
            n_changes_positive=dataset_params['n_changes_positive'],
            n_changes_negative=dataset_params['n_changes_negative'],
            dataset_size=validation_dataset_size
        )
    else:
        raise ValueError('Unknown problem type: %s' % config['data']['problem'])
    return training_set, validation_set


def get_graph(batch):
    if len(batch) != 2:
        # if isinstance(batch, GraphData):
        graph = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        return node_features, edge_features, from_idx, to_idx, graph_idx
    else:
        graph, labels = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        labels = torch.from_numpy(labels).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx, labels
