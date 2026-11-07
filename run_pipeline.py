import argparse
from pathlib import Path
from src.download import download_data
from src.qa import run_qa

def run_pipeline(config_path):
    download_data(config_path)

    run_qa(config_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str)
    args = parser.parse_args()
    config_path = Path(args.config)
    
    run_pipeline(config_path)