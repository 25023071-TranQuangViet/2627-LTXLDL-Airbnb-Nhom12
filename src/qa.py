import pandas as pd
from pathlib import Path
import requests
import yaml


def cleaning_data(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    pass