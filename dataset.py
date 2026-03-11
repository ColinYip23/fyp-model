import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gaussian_distance(distances: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """
    Convert scalar distances into a Gaussian basis expansion.

    Args:
        distances: Array of shape (n_edges,)
        centers: Gaussian centers of shape (n_centers,)
        width: Gaussian width

    Returns:
        Array of shape (n_edges, n_centers)
    """
    diff = distances[:, None] - centers[None, :]
    return np.exp(-(diff ** 2) / (width ** 2))


class GNoMEDataset(Dataset):
    """
    PyTorch Geometric dataset for GNoME crystal structures.

    Each graph:
    - nodes: atoms
    - node features: atomic number
    - edges: neighbor pairs within cutoff radius
    - edge features: Gaussian-expanded distances
    - target: formation energy per atom
    """

    def __init__(
        self,
        csv_path: str,
        cif_dir: str,
        n_samples: int = 10000,
        cutoff: float = 6.0,
        max_neighbors: int = 12,
        radius_gaussians: int = 50,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(None, transform, pre_transform)
        self.csv_path = csv_path
        self.cif_dir = cif_dir
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.seed = seed

        self.gaussian_centers = np.linspace(0, cutoff, radius_gaussians)
        self.gaussian_width = self.gaussian_centers[1] - self.gaussian_centers[0] if radius_gaussians > 1 else 0.2

        df = pd.read_csv(csv_path)

        # Adjust these column names if your CSV differs.
        self.id_col = "material_id"
        self.target_col = "formation_energy_per_atom"

        required = [self.id_col, self.target_col]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.dropna(subset=[self.id_col, self.target_col]).copy()

        # Keep only rows with existing CIF files.
        df["cif_path"] = df[self.id_col].astype(str).apply(lambda mid: os.path.join(cif_dir, f"{mid}.cif"))
        df = df[df["cif_path"].apply(os.path.exists)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No valid rows found with matching CIF files.")

        # Sample subset for initial experiment.
        set_seed(seed)
        if n_samples < len(df):
            df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

        self.df = df.reset_index(drop=True)

    def len(self) -> int:
        return len(self.df)

    def get(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        cif_path = row["cif_path"]
        y = float(row[self.target_col])

        structure = Structure.from_file(cif_path)

        atomic_numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)
        x = torch.tensor(atomic_numbers, dtype=torch.long).view(-1, 1)

        all_src = []
        all_dst = []
        all_dist = []

        for i, site in enumerate(structure):
            neighbors = structure.get_neighbors(site, self.cutoff)
            neighbors = sorted(neighbors, key=lambda nn: nn.nn_distance)[: self.max_neighbors]

            for nn in neighbors:
                j = nn.index
                d = float(nn.nn_distance)
                all_src.append(i)
                all_dst.append(j)
                all_dist.append(d)

        if len(all_src) == 0:
            # fallback for rare edge-less cases
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(self.gaussian_centers)), dtype=torch.float)
        else:
            edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
            dist_array = np.array(all_dist, dtype=np.float32)
            edge_features = gaussian_distance(dist_array, self.gaussian_centers, self.gaussian_width)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([y], dtype=torch.float),
            num_nodes=len(atomic_numbers),
        )
        return data