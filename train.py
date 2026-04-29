import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import GNoMEDataset
from model import CGCNNRegressorStrong


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_targets(dataset) -> np.ndarray:
    targets = []
    for data in dataset:
        y_val = data.y.view(-1).cpu().numpy()[0]
        targets.append(y_val)
    return np.array(targets, dtype=np.float32)


def compute_target_stats(train_dataset) -> Tuple[float, float]:
    train_targets = get_targets(train_dataset)
    mean = float(train_targets.mean())
    std = float(train_targets.std())

    if std < 1e-8:
        std = 1.0

    return mean, std


def normalize_targets(dataset, mean: float, std: float):
    for data in dataset:
        data.y = (data.y - mean) / std
    return dataset


def denormalize(values, mean: float, std: float):
    return values * std + mean


def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)

            pred = out.view(-1).detach().cpu().numpy()
            true = batch.y.view(-1).detach().cpu().numpy()

            pred = denormalize(pred, target_mean, target_std)
            true = denormalize(true, target_mean, target_std)

            preds.extend(pred.tolist())
            targets.extend(true.tolist())

    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds) ** 0.5
    r2 = r2_score(targets, preds)

    return mae, rmse, r2


def main():
    set_seed(42)

    csv_path = "data/mp_summary_balanced.csv"
    cif_dir = "data/structures"
    save_path = "model_config_2.pt"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not os.path.isdir(cif_dir):
        raise FileNotFoundError(f"CIF directory not found: {cif_dir}")

    metadata_df = pd.read_csv(csv_path)

    if "label" not in metadata_df.columns:
        raise ValueError("CSV must contain a 'label' column for stratified splitting.")

    if "energy_above_hull" not in metadata_df.columns:
        raise ValueError("CSV must contain 'energy_above_hull' as the regression target.")

    dataset = GNoMEDataset(
        csv_path=csv_path,
        cif_dir=cif_dir,
        n_samples=None,
        cutoff=6.0,
        max_neighbors=12,
        radius_gaussians=50,
        seed=42,
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your CSV columns and CIF filenames.")

    print(f"Total valid samples: {len(dataset)}")

    # =========================
    # Stratified Train / Val / Test split
    # =========================
    indices = np.arange(len(dataset))

    # Important:
    # This assumes GNoMEDataset keeps the same row order as the CSV
    # after applying n_samples.
    labels = metadata_df["label"].values[:len(dataset)]

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    temp_labels = labels[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )

    train_dataset = dataset.index_select(train_idx.tolist())
    val_dataset = dataset.index_select(val_idx.tolist())
    test_dataset = dataset.index_select(test_idx.tolist())

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    print("Label balance:")
    print("Train:", np.bincount(labels[train_idx].astype(int)))
    print("Val:  ", np.bincount(labels[val_idx].astype(int)))
    print("Test: ", np.bincount(labels[test_idx].astype(int)))

    # =========================
    # Target normalization
    # =========================
    target_mean, target_std = compute_target_stats(train_dataset)

    print(f"Target mean (train only): {target_mean:.6f}")
    print(f"Target std  (train only): {target_std:.6f}")

    train_dataset = normalize_targets(train_dataset, target_mean, target_std)
    val_dataset = normalize_targets(val_dataset, target_mean, target_std)
    test_dataset = normalize_targets(test_dataset, target_mean, target_std)

    # =========================
    # DataLoaders
    # =========================
    batch_size = 32
    num_workers = 4 if os.name != "nt" else 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CGCNNRegressorStrong(
        num_embeddings=100,
        atom_emb_dim=64,
        edge_dim=50,
        hidden_dim=128,
        num_conv_layers=4,
        dropout=0.1,
    ).to(device)

    print("Model device:", next(model.parameters()).device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    criterion = nn.L1Loss()

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = 100
    patience = 15
    wait = 0
    best_val_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            target = batch.y.view(-1)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch).view(-1)
                loss = criterion(out, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch.num_graphs

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = running_loss / len(train_dataset)

        val_mae, val_rmse, val_r2 = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )

        scheduler.step(val_mae)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss (norm MAE): {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val R2: {val_r2:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            wait = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae": best_val_mae,
                "target_mean": target_mean,
                "target_std": target_std,
                "model_config": {
                    "num_embeddings": 100,
                    "atom_emb_dim": 64,
                    "edge_dim": 50,
                    "hidden_dim": 128,
                    "num_conv_layers": 4,
                    "dropout": 0.1,
                },
            }

            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")

        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print("\nLoading best checkpoint for final test...")

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_mae, test_rmse, test_r2 = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        target_mean=checkpoint["target_mean"],
        target_std=checkpoint["target_std"],
    )

    print("\nFinal Test Results")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2:   {test_r2:.4f}")


if __name__ == "__main__":
    main()