import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
from torch import nn
import typing as tp
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader


import plotly.express as px


from torch.amp import GradScaler, autocast
from features import create_features, normalize, create_time_features, create_all_normal_features

from functools import partial

from json import dumps

from functools import cache


import os


class LinearEnsemble(nn.Module):
    def __init__(self,
                 dataset_size: int,
                 n_input_features: int,
                 n_output_features: int,
                 edges: np.ndarray = None,  # omitted in this model
                 graph: np.ndarray = None,
                 shared_weights: bool = False,
                 ):
        super().__init__()
        self._n_independent_models = dataset_size if not shared_weights else 1

        self.W = nn.Parameter(
            data=torch.randn(size=(self._n_independent_models, n_output_features, n_input_features)),
            requires_grad=True,
        )

        self.bias = nn.Parameter(
            data=torch.zeros(self._n_independent_models, n_output_features),
            requires_grad=True,
        )

    def forward(self, X):
        hadamard_product = self.W * X.unsqueeze(2)  # emulates linear layer independent for every node
        
        return hadamard_product.sum(-1) + self.bias  # reduce (aggregate) element-wise products and sum with bias


def validate(model, val_loader, loss_fn):
    model.eval()

    losses = []
    for x, y in val_loader:
        with autocast(device, dtype=torch.float16):
            y_pred = model(x.to(device))
            loss = loss_fn(y_pred, y.to(device))

        losses.append(loss.detach())
    losses = np.mean(torch.tensor(losses).cpu().numpy().tolist())
    return losses

def train_and_validate(model, train_loader, val_loader, lr = 0.1, n_epochs: int = 500, weight_decay: float = 1e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.L1Loss()
    grad_scaler = GradScaler(device=device)

    val_metrics = []
    losses = []
    for i in tqdm(range(n_epochs)):
        model.train()
        for x, y in train_loader:
            with autocast(device, dtype=torch.float16):
                y_pred = model(x.to(device))
                # print(y_pred.shape, y.shape)
                loss = loss_fn(y_pred, y.to(device))

            grad_scaler.scale(loss).backward()
            losses.append(loss.detach())

            # grad_scaler.unscale_(optimizer)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
            if (i % 50) == 0:
                val_metrics.append(validate(model, val_loader=val_loader, loss_fn=loss_fn))

    losses = torch.tensor(losses).cpu().numpy().tolist()
    val_metric = min(val_metrics)
    val_metric = min(val_metric, validate(model, val_loader=val_loader, loss_fn=loss_fn))
    return losses, val_metric



class MLP(nn.Sequential):
    def __init__(self,
                 dataset_size: int,
                 n_input_features: int,
                 n_output_features: int,
                 edges: np.ndarray = None,  # omitted in this model
                 graph: np.ndarray = None,
                 shared_weights: bool = False,
):
        super().__init__()
        self.mlp1 = LinearEnsemble(dataset_size=dataset_size, n_input_features=n_input_features, n_output_features=n_output_features, shared_weights=shared_weights)
        self.relu = nn.ReLU()
        self.mlp2 = LinearEnsemble(dataset_size=dataset_size, n_input_features=n_output_features, n_output_features=n_output_features, shared_weights=shared_weights)
        
        
class GraphModel(nn.Module):
    def __init__(self,
                 dataset_size: int,
                 n_input_features: int,
                 n_output_features: int,
                 edges: np.ndarray = None,
                 shared_weights: bool = False,
                 n_layers: int = 1,
                 aggr_mode: tp.Literal["MeanAggr", "GCN"] = "MeanAggr",
        ):

        super().__init__()
        adj_matrix = self._construct_adjacency_matrix(number_of_nodes=dataset_size, edgelist=edges).float()
        if aggr_mode == "MeanAggr":
            node_degrees = torch.sum(adj_matrix, 1)
            node_degrees = torch.where(node_degrees == 0, 1, node_degrees)

            degree_normalization_matrix = (1/node_degrees * np.eye(dataset_size)).float()
            graph_aggregation_matrix = degree_normalization_matrix @ adj_matrix
        elif aggr_mode == "GCN":
            adj_matrix = (adj_matrix + np.eye(dataset_size)).float()
            node_degrees = torch.sum(adj_matrix, 1)
            degree_normalization_matrix = (1/torch.sqrt(node_degrees) * np.eye(dataset_size)).float()
            graph_aggregation_matrix = degree_normalization_matrix @ adj_matrix @ degree_normalization_matrix

        else:
            raise ValueError(f"Aggregation mode `{aggr_mode}` isn't supported")

        
        
        self.register_buffer(name="graph_aggregation_matrix", tensor=graph_aggregation_matrix.float())

        self.linear_models = nn.ModuleList(
            [MLP(dataset_size=dataset_size, n_input_features=n_input_features, n_output_features=n_output_features, shared_weights=shared_weights)] + [
            MLP(dataset_size=dataset_size, n_input_features=2 * n_output_features, n_output_features=n_output_features, shared_weights=shared_weights)
            for _ in range(n_layers-1)
        ])
        
        self.n_layers = n_layers
        self.final_layer = nn.Linear(in_features=2 * n_output_features, out_features=n_output_features)

    def _graph_aggregation(self, h):
        return torch.cat([self.graph_aggregation_matrix @ h, h], -1)

    @staticmethod
    def _construct_adjacency_matrix(number_of_nodes: int, edgelist: np.ndarray):
        adjacency_matrix = torch.zeros((number_of_nodes, number_of_nodes))
        for u, v in edgelist:
            adjacency_matrix[u, v] = 1.0
        return adjacency_matrix

    def forward(self, x):
        # print(x.shape)
        h = self.linear_models[0](x)
        x = self._graph_aggregation(h)

        # print(h.shape)
        # print(x.shape)
        for layer in self.linear_models[1:]:

            h = layer(x)

            x = self._graph_aggregation(h) + x


        x = self.final_layer(x)
        return x


data = np.load('../data/metr_la_new.npz', allow_pickle=True)
list(data.keys())

dataset = data['targets']
dataset_size = len(dataset)

for i in range(0, len(dataset)):
    for j in range(0, len(dataset[i])):
        if np.isnan(dataset[i][j]) and i == 0:
            dataset[i][j] = 0
        if np.isnan(dataset[i][j]):
            dataset[i][j] = dataset[max(i - 1, 0)][j]


train_size = int(0.6 * dataset_size)
test_size =  int(0.2 * dataset_size)
def dataset_for_vertice(vertice):
    return dataset[:, vertice]
coef = 6
pred_cnt = 12
device = os.environ["DEVICE"] #"cuda:4"


## Dataset preparation

import random
samples_train = [i for i in random.sample(range(coef, int(dataset_size * 0.7)), train_size)]

X_train = [[dataset_for_vertice(j)[i - coef: i] for j in range(0, 207)] for i in samples_train]
X_train = np.array(X_train)
#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

y_train = [[dataset_for_vertice(j)[i:i + pred_cnt] for j in range(0,207)] for i in samples_train]
y_train = np.array(y_train)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1] * y_train.shape[2])

samples_test = [i for i in random.sample(range(int(dataset_size * 0.7), dataset_size - pred_cnt), test_size)]
X_test = [[dataset_for_vertice(j)[i - coef: i] for j in range(0, 207)] for i in samples_test]
X_test = np.array(X_test)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

y_test = [[dataset_for_vertice(j)[i:i + pred_cnt] for j in range(0,207)] for i in samples_test]
y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1] * y_test.shape[2])


graph = data['edges_pruned_by_partial_correlation']
graph = data['edges']

graph_view = np.zeros((207, 207))

for node_1, node_2 in graph:
    graph_view[node_2][node_1] = 1


time_features_expanded = create_time_features(
    timestamps=pd.date_range(start=data["first_timestamp_datetime"].item(),
                             end=data["last_timestamp_datetime"].item(),
                             freq="5min",
                           ),
    unix_timeseconds=data["unix_timestamps"],
    size_of_timestamps=data["targets"].shape[0]
)[:, None, :].repeat(data["targets"].shape[1], 1)


@cache
def prepare_data(
    mode: tp.Literal["default", "add-features","graph-aggr-default", "graph-aggr-features"],
    ):
    if mode == "default":
        X_tr = normalize(X_train)
        # X_test_normal_features = normalize(X_test_with_features)
        X_te = normalize(X_test)

    elif mode == "add-features":
        X_all_normal_features = create_all_normal_features(list([i,i] for i in range(207)),
                                                                 mode = 'max')
        X_all_normal_features_train = X_all_normal_features[samples_train]
        X_all_normal_features_test = X_all_normal_features[samples_test]

        X_tr = np.concatenate([normalize(X_train), X_all_normal_features_train], 2)
        X_te = np.concatenate([normalize(X_test), X_all_normal_features_test], 2)

    elif mode == "graph-aggr-features":
        X_all_normal_features = create_all_normal_features(
        graph.tolist(),
        mode = ["mean"])

        X_all_normal_features_train = X_all_normal_features[samples_train]
        X_all_normal_features_test = X_all_normal_features[samples_test]

        X_tr = np.concatenate((normalize(X_train), X_all_normal_features_train), axis= 2)
        # X_test_normal_features = normalize(X_test_with_features)
        X_te = np.concatenate((normalize(X_test), X_all_normal_features_test), axis= 2)

    X_train_torch = torch.from_numpy(X_tr).float()
    X_test_torch = torch.from_numpy(X_te).float()
    print(f"{X_train_torch.shape=} {X_test_torch}")
    
    Y_train_ensemble = torch.tensor([[dataset_for_vertice(j)[i:i + pred_cnt] for j in range(0,207)] for i in samples_train]).float()
    Y_test_ensemble = torch.tensor([[dataset_for_vertice(j)[i:i + pred_cnt] for j in range(0,207)] for i in samples_test]).float()

    ensemble_dataset_train = TensorDataset(X_train_torch, Y_train_ensemble)
    ensemble_dataset_test = TensorDataset(X_test_torch, Y_test_ensemble)

    dataloader_train_ensemble = DataLoader(ensemble_dataset_train, shuffle = True, batch_size = 1024)
    dataloader_test_ensemble = DataLoader(ensemble_dataset_test, shuffle = False, batch_size = 1024)


    return X_train_torch, Y_train_ensemble, dataloader_train_ensemble, dataloader_test_ensemble


def log_results(parameters):
    with open(f"{os.environ['MODEL']}_results.txt", "a") as f_write:
        print(dumps(parameters), file=f_write)


MODELS = ["GNN-Mean", "GNN-GCN", "LINEAR", "LINEAR-SHARED"]
LR = [1e-2, 1e-3, 0.1]
N_LAYERS = [1, 2]


MODELS = [os.environ["MODEL"]]

for data_mode in ["graph-aggr-features"]:
    X_train, Y_train_ensemble, train_loader, val_loader = prepare_data(data_mode)

    for n_layers in N_LAYERS:
        for model in MODELS:
            if model in ["LINEAR", "LINEAR-SHARED"] and n_layers != 1:
                continue

            for lr in LR:
                if model == "GNN-Mean":
                    model_callable = partial(GraphModel, shared_weights=True, edges=graph, n_layers=2)
                elif model == "GNN-GCN":
                    model_callable = partial(GraphModel, aggr_mode="GCN", shared_weights=True, edges=graph, n_layers=2)
                elif model == "LINEAR":
                    model_callable = partial(LinearEnsemble, shared_weights=False)
                elif model == "LINEAR-SHARED":
                    model_callable = partial(LinearEnsemble, shared_weights=True)

                model_callable = model_callable(dataset_size=X_train.shape[1], n_input_features=X_train.shape[2], n_output_features=Y_train_ensemble.shape[2]).to(device)
                _, val_metric = train_and_validate(model_callable, train_loader, val_loader, lr=lr, n_epochs=100, weight_decay=0)

                log_dict = dict(
                    model=model,
                    data_mode=data_mode,
                    n_layers=n_layers,
                    lr=lr,
                    val_metric=val_metric,
                )
                print(f"DONE: {log_dict}")
                log_results(log_dict)
