import pandas as pd
from pathlib import Path
import requests
import yaml


def cleaning_data(config_path):
        with Path(config_path).open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        for date in config["snapshots"]:
                #xu ly listing 
                listing_path = Path(f"data/raw/{date}/listings.csv.gz")
                listing = pd.read_csv(listing_path, compression= "gzip", dtype = {"id": "str"})
                listing_loc = pd.read_csv(Path("data/processed/listings-loc.csv"), dtype = {"listing_id": "str"}, usecols = ["listing_id", "analysis_area"])
                vung_loc = pd.read_csv(Path("data/processed/vung-loc.csv"))
                vung_loc["lon_min"] = pd.to_numeric(vung_loc["lon_min"].str.replace(",", ".", regex = False))
                vung_loc["lat_min"] = pd.to_numeric(vung_loc["lat_min"].str.replace(",", ".", regex = False))
                vung_loc["lon_max"] = pd.to_numeric(vung_loc["lon_max"].str.replace(",", ".", regex = False))
                vung_loc["lat_max"] = pd.to_numeric(vung_loc["lat_max"].str.replace(",", ".", regex = False))
                ls_ghep = listing.merge(listing_loc, left_on = "id", right_on = "listing_id", how = "right").merge(vung_loc, on = "analysis_area", how = "left")
                #xu ly tiep listing
                
                #xu ly review
                review_path = Path(f"data/raw/{date}/reviews.csv.gz")
                reviews = pd.read_csv(review_path, compression= "gzip", dtype = {"listing_id": "str", "id": "str", "reviewer_id": "str"}, parse_dates = ["date"])
                review_loc = pd.read_csv(Path("data/processed/reviews-loc.csv"), usecols = ["listing_id", "review_id"], dtype = {"listing_id": "str", "review_id": "str"})
                review_ghep = reviews.merge(review_loc, left_on = "id", right_on = "review_id", how = "right")
                #xu ly tiep review
                
                
                #xu ly calendar
                calendar_path = Path(f"data/raw/{date}/calendar.csv.gz")
                calendar = pd.read_csv(calendar_path, compression='gzip', dtype = {"listing_id": "str"}, parse_dates = ["date"])
                calendar_ghep = calendar.merge(listing_loc[["listing_id"]], on = "listing_id", how = "right")
                calendar_ghep["available"] = calendar_ghep["available"].map({"t": True, "f": False})
                #xu ly tiep calendar
                
                
                #xuat du lieu da xu ly
                processed_path = Path(f"data/processed/{date}")
                processed_path.mkdir(parents = True, exist_ok = True)
                clean_listing = Path(f"data/processed/{date}/listing_cleaned.csv")
                ls_ghep.to_csv(clean_listing, index = False)
                clean_review = Path(f"data/processed/{date}/review_cleaned.csv")
                review_ghep.to_csv(clean_review, index = False)
                clean_calendar = Path(f"data/processed/{date}/calendar_cleaned.csv")
                calendar_ghep.to_csv(clean_calendar, index = False)