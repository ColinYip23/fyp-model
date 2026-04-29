"""
Fetch structures and properties from OQMD via the OPTIMADE /structures endpoint.
Saves CIF files and a CSV suitable for inference and accuracy evaluation.
"""
import os
import requests
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure, Lattice

OUT_DIR = "data"
STRUCT_DIR = os.path.join(OUT_DIR, "oqmd_optimade_structures")
CSV_PATH = os.path.join(OUT_DIR, "oqmd_optimade.csv")

os.makedirs(STRUCT_DIR, exist_ok=True)

BASE_URL = "https://oqmd.org/optimade/structures"
PAGE_LIMIT = 50
FILTER = "_oqmd_stability=0.0"

print("Fetching OQMD OPTIMADE structures...")
print(f"URL: {BASE_URL}")
print(f"page_limit: {PAGE_LIMIT}, filter: {FILTER}\n")

params = {
    "page_limit": PAGE_LIMIT,
    "filter": FILTER,
}

try:
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    entries = data.get("data", [])

    if not entries:
        raise ValueError("No structures returned from the OPTIMADE endpoint.")

    rows = []
    for entry in tqdm(entries, desc="Parsing structures"):
        attributes = entry.get("attributes", {})
        oqmd_entry_id = attributes.get("_oqmd_entry_id") or entry.get("id")
        material_id = f"oqmd-{oqmd_entry_id}"
        band_gap = attributes.get("_oqmd_band_gap")
        formation_energy = attributes.get("_oqmd_delta_e")
        formula = attributes.get("chemical_formula_reduced")
        nsites = attributes.get("nsites")

        lattice_vectors = attributes.get("lattice_vectors")
        positions = attributes.get("cartesian_site_positions")
        species = attributes.get("species_at_sites")

        if not lattice_vectors or not positions or not species:
            tqdm.write(f"Skipping entry {oqmd_entry_id}: missing structure data")
            continue

        try:
            lattice = Lattice(lattice_vectors)
            structure = Structure(
                lattice,
                species,
                positions,
                coords_are_cartesian=True,
            )

            cif_path = os.path.join(STRUCT_DIR, f"{material_id}.cif")
            structure.to(filename=cif_path)

            rows.append({
                "material_id": material_id,
                "oqmd_entry_id": oqmd_entry_id,
                "formula": formula,
                "nsites": nsites,
                "band_gap": band_gap,
                "formation_energy_per_atom": formation_energy,
            })
        except Exception as exc:
            tqdm.write(f"Failed to save structure for {oqmd_entry_id}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    print(f"\nSaved {len(df)} structures to {STRUCT_DIR}")
    print(f"Saved metadata CSV to {CSV_PATH}")
    print("\nYou can now run prediction and evaluation with predict.py.")

except requests.exceptions.RequestException as exc:
    print(f"Request error: {exc}")
    raise
except Exception as exc:
    print(f"Unexpected error: {exc}")
    raise
