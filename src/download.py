import requests
from pathlib import Path
import pandas as pd
import yaml

config_path = "configs/London.yaml"

def download_data(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    file_types = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]
    raw = Path("data/raw")
    raw.mkdir(parents = True, exist_ok = True)
    for date in config["snapshots"]:
       date_path = Path(f"data/raw/{date}")
       date_path.mkdir(parents = True, exist_ok = True)
       for type in file_types:
            url = (f"https://data.insideairbnb.com/{config["data_root"]}/{date}/data/{type}")
            file_path = Path(f"data/raw/{date}/{type}")
            if not file_path.exists():
                r = requests.get(url)
                r.raise_for_status()
                file_path.write_bytes(r.content)

if __name__ == "__main__":
    download_data(config_path)