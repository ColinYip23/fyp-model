import os
import pandas as pd
from mp_api.client import MPRester
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("MP_API_KEY")
OUT_DIR = "data"
STRUCT_DIR = os.path.join(OUT_DIR, "structures")
CSV_PATH = os.path.join(OUT_DIR, "mp_summary.csv")

os.makedirs(STRUCT_DIR, exist_ok=True)

rows = []
TOTAL_GOAL = 100000

with MPRester(API_KEY) as mpr:
    # 1. We remove 'limit' here. 
    # 2. We use search which returns a generator/iterator.
    docs_generator = mpr.materials.summary.search(
        is_stable=True,
        num_sites=(1, 50),
        fields=[
            "material_id",
            "structure",
            "formation_energy_per_atom",
            "nsites",
        ]
    )

    print(f"Fetching materials (Goal: {TOTAL_GOAL})...")
    
    # Use enumerate to keep track of how many we've processed
    for i, doc in enumerate(tqdm(docs_generator, total=TOTAL_GOAL)):
        if i >= TOTAL_GOAL:
            break
            
        material_id = str(doc.material_id)
        
        # Check if file already exists (Resume capability)
        cif_path = os.path.join(STRUCT_DIR, f"{material_id}.cif")
        if not os.path.exists(cif_path):
            doc.structure.to(filename=cif_path)

        rows.append({
            "material_id": material_id,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "nsites": doc.nsites,
        })

        # Save metadata every 1000 items so you don't lose progress
        if (i + 1) % 1000 == 0:
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

# Final Save
df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)
print(f"Finished! Total materials collected: {len(df)}")