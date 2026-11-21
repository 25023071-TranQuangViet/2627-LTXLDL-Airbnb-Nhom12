# 2627-LTXLDL-Airbnb-Nhom12

## Tạo môi trường ảo venv:
```powershell
    python -m venv .venv
    .venv\Scripts\activate
```

## Cài lại thư viện cần thiết trong requirement.txt:

```python -m pip install -r requirements.txt```

## Cài khóa API (chỉ cần khi chạy LLM)

Pipeline gọi Gemini qua biến môi trường `GEMINI_API_KEY`.

- Windows (PowerShell): `$env:GEMINI_API_KEY="khoa-cua-ban"`
- Linux / macOS: `export GEMINI_API_KEY="khoa-cua-ban"`

## Lệnh chạy:

- Có chạy llm:
```python run_pipeline.py --config configs/London.yaml```
- Không chạy llm:
```python run_pipeline.py --config configs/London.yaml --skip-llm```
