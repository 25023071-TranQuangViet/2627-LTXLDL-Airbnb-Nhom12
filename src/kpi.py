"""Mô-đun điều phối tính toán toàn bộ KPI của dự án (K1 đến K6)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fix ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kpi_listings import calculate_listings_kpis


def calculate_kpis(config_path: str | Path) -> None:
    print("--> [KPI Pipeline] Bắt đầu tính toán toàn bộ KPI dự án Inside Airbnb...")
    
    # 1. Tính toán nhóm KPI từ bảng listings (K1, K2, K3, K4 + sinh kpi_definitions.csv)
    results = calculate_listings_kpis(config_path)
    print("    -> [Xong] Đã hoàn thành các KPI từ bảng listings (K1 - K4).")

    # 2. Tích hợp KPI từ calendar (K5) - Chờ người làm module Calendar
    # 3. Tích hợp KPI từ kết quả LLM (K6) - Chờ người làm module LLM

    print("--> [KPI Pipeline] Toàn bộ các bảng KPI đã được ghi nhận vào data/processed/ và reports/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Điều phối tính toán KPI")
    parser.add_argument("--config", default="configs/London.yaml")
    args = parser.parse_args()
    calculate_kpis(args.config)