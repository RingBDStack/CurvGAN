import math
import random
import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
from tqdm import tqdm
import pandas as pd

data_path = './data'

def encode_onehot(labels, config):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    print("Num of classes: {}".format(len(classes)))
    config.num_classes = len(classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = list(map(classes_dict.get, labels))# np.array(, dtype=np.int32)
    labels_index = list(list(i).index(1) for i in labels_onehot)
    return np.array(labels_index, dtype=np.int32)


def read_graph(config):
    if config.dataset in ['cora', 'citeseer', 'webkb']:
        idx_features_labels = np.genfromtxt("{}/{}/{}.content".format(data_path, config.dataset, config.dataset), dtype=np.dtype(str))
        labels = encode_onehot(idx_features_labels[:, -1], config)
        idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/{}/{}.cites".format(data_path, config.dataset, config.dataset), dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        config.num_nodes = len(idx_map)
    elif config.dataset in ['polblogs']:
        edges = np.array(pd.read_csv("{}/{}/adjacency.csv".format(data_path, config.dataset), header=None))
        labels = np.array(pd.read_csv("{}/{}/labels.csv".format(data_path, config.dataset), header=None)).T.squeeze()
        classes = sorted(list(set(labels)))
        print("Num of classes: {}".format(len(classes)))
        config.num_classes = len(classes)
        config.num_nodes = labels.shape[0]
    elif config.dataset in ['ba', 'sbm', 'ws']:
        edges = torch.load("{}/{}/{}_edge_index.pt".format(data_path, config.dataset, config.dataset)).data.numpy().T
        try:
            labels = torch.load("{}/{}/{}_labels.pt".format(data_path, config.dataset, config.dataset)).data.numpy().T
            classes = sorted(list(set(labels)))
            print("Num of classes: {}".format(len(classes)))
            config.num_classes = len(classes)
        except:
            labels = None
        features = torch.load("{}/{}/{}_features.pt".format(data_path, config.dataset, config.dataset))
        config.num_nodes = features.shape[0]
    print("Num of nodes: {}".format(config.num_nodes))
    print("Num of edges: {}".format(len(edges)))
    return edges, labels


def split_edges(edges, config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    edges_copy = edges.copy()
    random.shuffle(edges_copy)
    threshold = math.ceil(len(edges) * config.test_prop)
    test_edges_pos, train_edges = edges_copy[:threshold], edges_copy[threshold:]
    graph = nx.Graph(edges_copy.tolist())
    # Set a max bound of test samples
    threshold = min(threshold, 5000)
    random.shuffle(test_edges_pos)
    test_edges_pos = test_edges_pos[:threshold]
    test_edges_neg = []
    with tqdm(total=threshold) as pbar:
        while len(test_edges_neg) < threshold:
            a, b = random.sample(range(config.num_nodes), 2)
            if a == b or graph.has_edge(a, b):
                continue
            else:
                test_edges_neg.append([a, b])
                pbar.update(1)
    return train_edges, np.array(test_edges_pos), np.array(test_edges_neg)


def load_data(config):
    print('Loading {} dataset...'.format(config.dataset))
    full_edges, labels = read_graph(config)
    if config.task in ['lp', 'nc']:
        train_edges, test_edges_pos, test_edges_neg = split_edges(full_edges, config)
        data = {
            "train_edges": train_edges,
            "train_nodes": list(set(train_edges.flatten())),
            "test_edges_pos": test_edges_pos.T,
            "test_edges_neg": test_edges_neg.T,
            "train_graph": nx.Graph(train_edges.tolist()),
            "labels": labels
        }
        return data
    if config.task in ['gr']:
        train_edges, _, _ = split_edges(full_edges, config)
        print(2571 in full_edges.flatten())
        data = {
            "train_edges": train_edges,
            "train_nodes": list(set(train_edges.flatten())),
            "test_nodes": set(random.sample(list(range(config.num_nodes)), math.ceil(0.1 * config.num_nodes))),
            "train_graph": nx.Graph(train_edges.tolist()),
            "full_graph": nx.Graph(full_edges.tolist())
        }
        return data


def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def read_embeddings(filename, n_node, n_embed):
    embedding_matrix = np.random.rand(n_node, n_embed)
    i = -1
    if filename[-3:] == 'pth':
        embedding_matrix = torch.load(filename, map_location='cpu')['embedding'].data
    else:
        with open(filename) as infile:
            for line in infile.readlines()[1:]:
                i += 1
                emd = line.strip().split()
                embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        embedding_matrix = torch.tensor(embedding_matrix)
    return embedding_matrix


