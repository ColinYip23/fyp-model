import torch
import pandas as pd
from torch_geometric.loader import DataLoader

from dataset import GNoMEDataset
from model import CGCNNRegressor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = GNoMEDataset(
        csv_path="data/inference.csv",
        cif_dir="data/inference_structures",
        n_samples=None,
        cutoff=6.0,
        max_neighbors=12,
        radius_gaussians=50,
        seed=42,
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = CGCNNRegressor(
        num_embeddings=100,
        atom_emb_dim=64,
        edge_dim=50,
        hidden_dim=128,
        num_conv_layers=4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load("best_cgcnn_mp_fe.pt", map_location=device))
    model.eval()

    predictions = []
    material_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            predictions.extend(out.detach().cpu().view(-1).tolist())
            material_ids.extend(batch.material_id)

    df = pd.DataFrame({
        "material_id": material_ids,
        "predicted_formation_energy_per_atom": predictions,
    })

    output_path = "data/inference_predictions.csv"
    df.to_csv(output_path, index=False)

    print("\nPredictions:")
    print(df)
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()