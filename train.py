import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import GNoMEDataset
from model import CGCNNRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.extend(out.detach().cpu().numpy().tolist())
            targets.extend(batch.y.view(-1).detach().cpu().numpy().tolist())

    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds) ** 0.5
    r2 = r2_score(targets, preds)
    return mae, rmse, r2


def main():
    set_seed(42)

    csv_path = "data/mp_summary.csv"
    cif_dir = "data/structures"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not os.path.isdir(cif_dir):
        raise FileNotFoundError(f"CIF directory not found: {cif_dir}")

    dataset = GNoMEDataset(
        csv_path=csv_path,
        cif_dir=cif_dir,
        n_samples=10000,
        cutoff=6.0,
        max_neighbors=12,
        radius_gaussians=50,
        seed=42,
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your CSV columns and CIF filenames.")

    print(f"Total valid samples: {len(dataset)}")

    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_dataset = dataset.index_select(train_idx.tolist())
    val_dataset = dataset.index_select(val_idx.tolist())
    test_dataset = dataset.index_select(test_idx.tolist())

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CGCNNRegressor(
        num_embeddings=100,
        atom_emb_dim=64,
        edge_dim=50,
        hidden_dim=128,
        num_conv_layers=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.L1Loss()

    best_val_mae = float("inf")
    save_path = "best_cgcnn_mp_fe.pt"

    epochs = 50
    patience = 10
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)
            target = batch.y.view(-1)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.num_graphs
            progress.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_dataset)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MAE: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val R2: {val_r2:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            wait = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print("\nLoading best checkpoint for final test...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_mae, test_rmse, test_r2 = evaluate(model, test_loader, device)
    print(
        f"Test MAE: {test_mae:.4f}\n"
        f"Test RMSE: {test_rmse:.4f}\n"
        f"Test R2: {test_r2:.4f}"
    )


if __name__ == "__main__":
    main()