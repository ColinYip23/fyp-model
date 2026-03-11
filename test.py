import os
import random

import numpy as np
import pandas as pd
import torch
from mp_api.client import MPRester
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

from dataset import GNoMEDataset
from model import CGCNNRegressor
from dotenv import load_dotenv

load_dotenv()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            preds.extend(out.detach().cpu().view(-1).tolist())
            targets.extend(batch.y.detach().cpu().view(-1).tolist())

    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds) ** 0.5
    r2 = r2_score(targets, preds)
    return mae, rmse, r2, preds, targets


def main():
    set_seed(42)

    api_key = os.getenv("MP_API_KEY")  # or hardcode temporarily
    if not api_key:
        raise ValueError("Set your Materials Project API key in MP_API_KEY")

    out_csv = "data/mp_random_50.csv"
    out_cif_dir = "data/mp_random_50_structures"
    checkpoint_path = "best_cgcnn_mp_fe.pt"

    os.makedirs("data", exist_ok=True)
    os.makedirs(out_cif_dir, exist_ok=True)

    # Optional but strongly recommended:
    # exclude materials already present in your training CSV
    train_csv = "data/mp_summary.csv"
    train_ids = set()
    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)
        if "material_id" in train_df.columns:
            train_ids = set(train_df["material_id"].astype(str).tolist())

    print("Downloading candidate materials from Materials Project...")

    with MPRester(api_key) as mpr:
        docs = list(
            mpr.materials.summary.search(
                is_stable=True,
                num_sites=(1, 50),
                fields=[
                    "material_id",
                    "structure",
                    "formation_energy_per_atom",
                ],
            )
        )

    if len(docs) == 0:
        raise ValueError("No materials returned from Materials Project.")

    # Keep only docs that have all needed fields and are not in training set
    valid_docs = []
    for d in docs:
        material_id = str(d.material_id)
        structure = d.structure
        fe = d.formation_energy_per_atom

        if material_id in train_ids:
            continue
        if structure is None or fe is None:
            continue

        valid_docs.append(d)

    if len(valid_docs) < 50:
        raise ValueError(
            f"Only found {len(valid_docs)} eligible materials after filtering. "
            "Try loosening filters or removing the train-set exclusion."
        )

    sampled_docs = random.sample(valid_docs, 50)

    rows = []
    for d in sampled_docs:
        material_id = str(d.material_id)
        structure = d.structure
        fe = float(d.formation_energy_per_atom)

        cif_path = os.path.join(out_cif_dir, f"{material_id}.cif")
        structure.to(filename=cif_path)

        rows.append(
            {
                "material_id": material_id,
                "formation_energy_per_atom": fe,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved CIFs to: {out_cif_dir}")
    print(f"Number of test materials: {len(df)}")

    dataset = GNoMEDataset(
        csv_path=out_csv,
        cif_dir=out_cif_dir,
        n_samples=None,
        cutoff=6.0,
        max_neighbors=12,
        radius_gaussians=50,
        seed=42,
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CGCNNRegressor(
        num_embeddings=100,   # keep exactly the same as training
        atom_emb_dim=64,
        edge_dim=50,
        hidden_dim=128,
        num_conv_layers=4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")

    mae, rmse, r2, preds, targets = evaluate(model, loader, device)

    print("\n=== Metrics on random 50 Materials Project materials ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    results_df = df.copy()
    results_df["predicted_formation_energy_per_atom"] = preds
    results_df["absolute_error"] = (
        results_df["formation_energy_per_atom"]
        - results_df["predicted_formation_energy_per_atom"]
    ).abs()

    results_path = "data/mp_random_50_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved predictions: {results_path}")
    print(results_df.head())


if __name__ == "__main__":
    main()