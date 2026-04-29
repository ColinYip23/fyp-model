import argparse
import os

import torch
import pandas as pd
from torch_geometric.loader import DataLoader

from dataset import GNoMEDataset
from model import CGCNNRegressor


def construct_parser():
    parser = argparse.ArgumentParser(description="Run model inference and optionally compute accuracy.")
    parser.add_argument("--csv", default="data/inference.csv", help="Path to input CSV file.")
    parser.add_argument("--cif-dir", default="data/inference_structures", help="Directory containing CIF files.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path. Defaults to best_cgcnn_mp_fe.pt or model_config.pt.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    return parser


def main():
    parser = construct_parser()
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        for candidate in ["best_cgcnn_mp_fe.pt", "model_config.pt"]:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Please provide --checkpoint or place best_cgcnn_mp_fe.pt/model_config.pt in the workspace."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Loading checkpoint: {checkpoint_path}")

    dataset = GNoMEDataset(
        csv_path=args.csv,
        cif_dir=args.cif_dir,
        n_samples=None,
        cutoff=6.0,
        max_neighbors=12,
        radius_gaussians=50,
        seed=42,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        model_config = checkpoint.get(
            "model_config",
            {
                "num_embeddings": 100,
                "atom_emb_dim": 64,
                "edge_dim": 50,
                "hidden_dim": 128,
                "num_conv_layers": 4,
                "dropout": 0.1,
            },
        )
    else:
        model_state = checkpoint
        model_config = {
            "num_embeddings": 100,
            "atom_emb_dim": 64,
            "edge_dim": 50,
            "hidden_dim": 128,
            "num_conv_layers": 4,
            "dropout": 0.1,
        }

    model = CGCNNRegressor(
        num_embeddings=model_config.get("num_embeddings", 100),
        atom_emb_dim=model_config.get("atom_emb_dim", 64),
        edge_dim=model_config.get("edge_dim", 50),
        hidden_dim=model_config.get("hidden_dim", 128),
        num_conv_layers=model_config.get("num_conv_layers", 4),
        dropout=model_config.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()

    predictions = []
    material_ids = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            predictions.extend(out.detach().cpu().view(-1).tolist())
            material_ids.extend(batch.material_id)
            if dataset.has_target:
                targets.extend(batch.y.detach().cpu().view(-1).tolist())

    df = pd.DataFrame({
        "material_id": material_ids,
        "predicted_formation_energy_per_atom": predictions,
    })

    if dataset.has_target:
        df["formation_energy_per_atom"] = targets
        df["absolute_error"] = (df["formation_energy_per_atom"] - df["predicted_formation_energy_per_atom"]).abs()
        mae = df["absolute_error"].mean()
        rmse = ((df["formation_energy_per_atom"] - df["predicted_formation_energy_per_atom"]) ** 2).mean() ** 0.5
        ss_res = ((df["formation_energy_per_atom"] - df["predicted_formation_energy_per_atom"]) ** 2).sum()
        ss_tot = ((df["formation_energy_per_atom"] - df["formation_energy_per_atom"].mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

        print("\nEvaluation metrics:")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

    output_path = os.path.splitext(args.csv)[0] + "_predictions.csv"
    df.to_csv(output_path, index=False)

    print("\nPredictions:")
    print(df.head(20))
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()