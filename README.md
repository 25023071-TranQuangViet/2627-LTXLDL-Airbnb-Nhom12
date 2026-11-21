# 2627-LTXLDL-Airbnb-Nhom12

## Tạo môi trường ảo venv:
```powershell
    python -m venv .venv
    .venv\Scripts\activate
```

## Cài lại thư viện cần thiết trong requirement.txt:

```python -m pip install -r requirements.txt```

## Lệnh chạy:

- Có chạy llm:
```python run_pipeline.py --config configs/London.yaml```
- Không chạy llm:
```python run_pipeline.py --config configs/London.yaml --skip-llm```

**Tên biến môi trường gọi API**: `GEMINI_API_KEY`