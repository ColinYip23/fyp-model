import os
import pandas as pd
from mp_api.client import MPRester
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MP_API_KEY")

OUT_DIR = "data"
STRUCT_DIR = os.path.join(OUT_DIR, "structures")
CSV_PATH = os.path.join(OUT_DIR, "mp_summary_balanced.csv")

os.makedirs(STRUCT_DIR, exist_ok=True)

# Balanced target
STABLE_GOAL = 50000
UNSTABLE_GOAL = 50000

rows = []


def save_structure(doc, label):
    material_id = str(doc.material_id)

    cif_path = os.path.join(STRUCT_DIR, f"{material_id}.cif")

    if not os.path.exists(cif_path):
        doc.structure.to(filename=cif_path)

    rows.append({
        "material_id": material_id,
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "energy_above_hull": doc.energy_above_hull,
        "nsites": doc.nsites,
        "is_stable": doc.is_stable,
        "label": label
    })


with MPRester(API_KEY) as mpr:

    fields = [
        "material_id",
        "structure",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
        "nsites",
    ]

    # -------------------------
    # Stable materials
    # energy_above_hull = 0
    # label = 1
    # -------------------------
    stable_docs = mpr.materials.summary.search(
        is_stable=True,
        num_sites=(1, 50),
        fields=fields
    )

    print(f"Fetching stable materials: {STABLE_GOAL}")

    for i, doc in enumerate(tqdm(stable_docs, total=STABLE_GOAL)):
        if i >= STABLE_GOAL:
            break

        save_structure(doc, label=1)

        if (i + 1) % 1000 == 0:
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    # -------------------------
    # Unstable / metastable materials
    # energy_above_hull > 0
    # label = 0
    # -------------------------
    unstable_docs = mpr.materials.summary.search(
        is_stable=False,
        energy_above_hull=(0.01, 1.0),
        num_sites=(1, 50),
        fields=fields
    )

    print(f"Fetching unstable materials: {UNSTABLE_GOAL}")

    for i, doc in enumerate(tqdm(unstable_docs, total=UNSTABLE_GOAL)):
        if i >= UNSTABLE_GOAL:
            break

        save_structure(doc, label=0)

        if (i + 1) % 1000 == 0:
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

# Final save
df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)

print(f"Finished! Total materials collected: {len(df)}")
print(df["label"].value_counts())