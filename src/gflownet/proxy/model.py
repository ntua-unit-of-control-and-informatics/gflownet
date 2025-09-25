import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import GATConv


class GraphAttention(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_dim,
        gnn_layers,
        gnn_channels,
        heads,
        dropout_proba,
        mlp_layers=1,
        mlp_channels=64,
    ):
        super(GraphAttention, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(
                in_channels=node_feats,
                out_channels=gnn_channels,
                heads=heads,
                dropout=dropout_proba,
                concat=True,
                edge_dim=edge_dim,
            )
        )
        for l in range(gnn_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=gnn_channels * heads,
                    out_channels=gnn_channels,
                    heads=heads,
                    dropout=dropout_proba,
                    concat=True,
                    edge_dim=edge_dim,
                )
            )
        self.gat_layers.append(
            GATConv(
                in_channels=gnn_channels * heads,
                out_channels=gnn_channels,
                heads=heads,
                dropout=dropout_proba,
                concat=False,
                edge_dim=edge_dim,
            )
        )
        self.dropout = nn.Dropout(dropout_proba)
        self.activation = nn.LeakyReLU()
        for l in range(gnn_layers):
            out_dim = gnn_channels * heads if l < gnn_layers - 1 else gnn_channels
        self.mlp = FullyConnected(
            gnn_channels,
            mlp_layers,
            mlp_channels,
            dropout_proba,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        for l, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x


class FullyConnected(nn.Module):
    def __init__(self, input_dim, layers, channels, dropout_proba):
        super(FullyConnected, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_dim, channels))
        for _ in range(layers - 2):
            self.fc_layers.append(nn.Linear(channels, channels))
        self.fc_layers.append(nn.Linear(channels, 1))
        self.dropout = nn.Dropout(dropout_proba)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if not i == len(self.fc_layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x


def load_proxy_to_gflow(param_file,path):
    args_dict = {}
    with open(param_file, "r") as f:
        for line in f:
            key, value = line.strip().split(": ", 1)  # split only at first ": "
            # Try to cast back to int/float if possible
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as string if not int/float
            args_dict[key] = value

    model = GraphAttention(
        node_feats=29,
        edge_dim=7,
        gnn_layers=args_dict["gnn_layers"],
        gnn_channels=args_dict["gnn_channels"],
        heads=args_dict["heads"],
        dropout_proba=args_dict["dropout_proba"],
        mlp_layers=args_dict["mlp_layers"],
        mlp_channels=args_dict["gnn_channels"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
