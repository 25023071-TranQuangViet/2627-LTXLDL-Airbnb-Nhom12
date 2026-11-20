import argparse
import os
from pathlib import Path
from src.download import download_data
from src.qa import run_qa
from src.kpi import tinh_kpi
from src.baseline import run_baseline
from src.llm import run_llm
from src.evaluate import run_evaluate
from src.viz import ve_tat_ca

def run_pipeline(config_path, skip_llm = False, llm_limit = None):
    download_data(config_path)

    run_qa(config_path)
    
    run_baseline(config_path)
    
    run_llm(config_path, skip_llm = skip_llm)
    
    run_evaluate(config_path)
    
    tinh_kpi(config_path)
    
    ve_tat_ca(config_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="chỉ gọi API cho tối đa N review mới")
    args = parser.parse_args()
    config_path = Path(args.config)
    
    run_pipeline(config_path, args.skip_llm, args.limit)