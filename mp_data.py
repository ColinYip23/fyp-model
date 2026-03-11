import os
import pandas as pd
from mp_api.client import MPRester

API_KEY = os.getenv("MP_API_KEY")
OUT_DIR = "data"
STRUCT_DIR = os.path.join(OUT_DIR, "structures")
CSV_PATH = os.path.join(OUT_DIR, "mp_summary.csv")

os.makedirs(STRUCT_DIR, exist_ok=True)

rows = []

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        is_stable=True,
        num_sites=(1, 50),
        fields=[
            "material_id",
            "structure",
            "formation_energy_per_atom",
            "nsites",
        ],
    )

    for doc in docs[:10000]:
        material_id = str(doc.material_id)
        structure = doc.structure

        cif_path = os.path.join(STRUCT_DIR, f"{material_id}.cif")
        structure.to(filename=cif_path)

        rows.append({
            "material_id": material_id,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "nsites": doc.nsites,
        })

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)

print(f"Saved {len(df)} rows to {CSV_PATH}")