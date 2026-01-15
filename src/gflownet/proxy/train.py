import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from gflownet.proxy.model import GraphAttention
from gflownet.proxy.mol_utils import scaffold_split, smiles2graph, random_split


def build_loaders(train_dataset, valid_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, valid_loader, test_loader


def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        x, edge_index, edge_attr, batch, y = data_from_loader(data, device)
        pred = model(x, edge_index, edge_attr, batch)
        loss = F.mse_loss(pred.squeeze(dim=-1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader)
    scheduler.step(total_loss)
    return total_loss


def eval_epoch(model, val_loader, device):
    model.eval()
    total_loss = 0
    for data in val_loader:
        x, edge_index, edge_attr, batch, y = data_from_loader(data, device)
        with torch.no_grad():
            pred = model(x, edge_index, edge_attr, batch)
        loss = F.mse_loss(pred.squeeze(dim=-1), y)
        total_loss += loss.item()
    return total_loss / len(val_loader)


def infer_model(model, loader, device):
    all_preds = []
    all_true = []
    for data in loader:
        x, edge_index, edge_attr, batch, y = data_from_loader(data, device)
        with torch.no_grad():
            pred = model(x, edge_index, edge_attr, batch)
        pred_list = [pr for pr in pred.squeeze(dim=-1).cpu().numpy()]
        all_preds.extend(pred_list)
        true_list = [tr for tr in y.cpu().numpy()]
        all_true.extend(true_list)
    return all_preds, all_true


def get_metrics(preds, true):
    rmse = root_mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    r2 = r2_score(true, preds)

    return rmse, mae, r2


def data_from_loader(data, device):
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    batch = data.batch.to(device)
    y = data.y.to(device)

    return x, edge_index, edge_attr, batch, y


if __name__ == "__main__":
    def parse_args():
        argparser = argparse.ArgumentParser(description="GNN for KOW prediction")
        argparser.add_argument("--learning_rate", type=float, default=0.001)
        argparser.add_argument("--batch_size", type=int, default=64)
        argparser.add_argument("--gnn_layers", type=int, default=2)
        argparser.add_argument("--gnn_channels", type=int, default=64)
        argparser.add_argument("--heads", type=int, default=4)
        argparser.add_argument("--mlp_layers", type=int, default=2)
        argparser.add_argument("--dropout_proba", type=float, default=0.2)
        argparser.add_argument(
            "--data_url",
            type=str,
            default=r"https://raw.githubusercontent.com/CesareWang/Predictors-for-15-Environmental-Endpoints/main/predictors/data/SW.csv",
        )
        argparser.add_argument("--target_col", type=str, default="logKOW")
        argparser.add_argument("--rename_from", type=str, default="active")
        argparser.add_argument("--rename_to", type=str, default="logKOW")
        argparser.add_argument("--params_out", type=str, default="model_params.txt")
        argparser.add_argument("--best_model_out", type=str, default="best_model.pt")
        return argparser.parse_args()

    args = parse_args()

    with open(args.params_out, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # ------------  Load data

    data_url = args.data_url
    kow_data = pd.read_csv(data_url, index_col=0)
    if args.rename_from and args.rename_to and args.rename_from in kow_data.columns:
        kow_data.rename(columns={args.rename_from: args.rename_to}, inplace=True)

    # ------------ Create data splits with scaffold
    train_id, valid_id, test_id = random_split(frac_train=0.80, smiles=kow_data)
    train_data = kow_data.loc[train_id]
    valid_data = kow_data.loc[valid_id]
    test_data = kow_data.loc[test_id]
    # ------------ Create GNN datasets
    train_dataset = [
        smiles2graph(Chem.MolFromSmiles(smiles), y)
        for smiles, y in zip(train_data["smiles"], train_data[args.target_col])
    ]
    valid_dataset = [
        smiles2graph(Chem.MolFromSmiles(smiles), y)
        for smiles, y in zip(valid_data["smiles"], valid_data[args.target_col])
    ]
    test_dataset = [
        smiles2graph(Chem.MolFromSmiles(smiles), y)
        for smiles, y in zip(test_data["smiles"], test_data[args.target_col])
    ]
    # ------------ Loaders
    tr_loader, val_loader, te_loader = build_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=args.batch_size
    )
    # ------------- Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAttention(
        node_feats=29,
        edge_dim=7,
        gnn_layers=args.gnn_layers,
        gnn_channels=args.gnn_channels,
        heads=args.heads,
        dropout_proba=args.dropout_proba,
        mlp_layers=args.mlp_layers,
        mlp_channels=args.gnn_channels,
    ).to(device)

    # --------------- Training configs
    epochs = 25
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)

    # --------------- Training
    best_loss = 1000
    for epoch in range(epochs):
        train_loss = train_epoch(model, tr_loader, optimizer, scheduler, device)
        val_loss = eval_epoch(model, val_loader, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{args.best_model_out}")
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    torch.save(model.state_dict(), "final_model.pt")
    # Inference
    all_preds, all_true = infer_model(model, te_loader, device)
    rmse, mae, r2 = get_metrics(all_preds, all_true)
    print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
