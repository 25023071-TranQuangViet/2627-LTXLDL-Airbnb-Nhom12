import requests
from pathlib import Path
import pandas as pd
    
data_file = ["listings", "calendar", "reviews"]
for i in data_file:
    i_2025_09_14 = Path(f"data/raw/{i}_2025-09-14.csv.gz")
    if not i_2025_09_14.exists():
        r_2025_09_14 = requests.get(("https://data.insideairbnb.com/united-kingdom/england/london/"f"2025-09-14/data/{i}.csv.gz"), timeout=120)
        r_2025_09_14.raise_for_status()
        i_2025_09_14.write_bytes(r_2025_09_14.content)
    
    i_2025_12_13 = Path(f"data/raw/{i}_2025-12-13.csv.gz")
    if not i_2025_12_13.exists():
        r_2025_12_13 = requests.get(("https://data.insideairbnb.com/united-kingdom/england/london/"f"2025-12-13/data/{i}.csv.gz"), timeout=120)
        r_2025_12_13.raise_for_status()
        i_2025_12_13.write_bytes(r_2025_12_13.content)
    
    i_2026_03_21 = Path(f"data/raw/{i}_2026-03-21.csv.gz")
    if not i_2026_03_21.exists():
        r_2026_03_21 = requests.get(("https://data.insideairbnb.com/united-kingdom/england/london/"f"2026-03-21/data/{i}.csv.gz"), timeout=120)
        r_2026_03_21.raise_for_status()
        i_2026_03_21.write_bytes(r_2026_03_21.content)
    
    i_2026_06_19 = Path(f"data/raw/{i}_2026-06-19.csv.gz")
    if not i_2026_06_19.exists():
        r_2026_06_19 = requests.get(("https://data.insideairbnb.com/united-kingdom/england/london/"f"2026-06-19/data/{i}.csv.gz"), timeout=120)
        r_2026_06_19.raise_for_status()
        i_2026_06_19.write_bytes(r_2026_06_19.content)
    
