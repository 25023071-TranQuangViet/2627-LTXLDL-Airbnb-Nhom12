# Bản nháp báo cáo kiểm tra dữ liệu - DSE2049-C04-G12-london

## 1. Tệp không tải được / không đọc được
- Không có (cả 3 tệp của 4 mốc đều tồn tại và đọc được).
  (Mã lỗi HTTP khi tải, ví dụ 403, lấy từ bước download.py; script này chỉ kiểm tệp đã có trên đĩa.)

## 2. Mẫu LLM và nhãn tay tìm được
- Mốc 2025-09-14: mẫu LLM 1000/1000, nhãn tay 100/100
- Mốc 2025-12-13: mẫu LLM 1000/1000, nhãn tay 100/100
- Mốc 2026-03-21: mẫu LLM 1000/1000, nhãn tay 100/100
- Mốc 2026-06-19: mẫu LLM 1000/1000, nhãn tay 100/100

## 3. Cột listings có từ 90% NaN trở lên
- Mốc 2025-09-14: calendar_updated, license, neighbourhood_group_cleansed
- Mốc 2025-12-13: calendar_updated, estimated_revenue_l365d, license, neighborhood_overview, neighbourhood, neighbourhood_group_cleansed, price
- Mốc 2026-03-21: calendar_updated, host_acceptance_rate, host_neighbourhood, host_response_rate, host_response_time, host_since, host_thumbnail_url, host_total_listings_count, host_verifications, instant_bookable, license, neighborhood_overview, neighbourhood, neighbourhood_group_cleansed
- Mốc 2026-06-19: calendar_updated, host_acceptance_rate, host_neighbourhood, host_response_rate, host_response_time, host_since, host_thumbnail_url, host_total_listings_count, host_verifications, instant_bookable, license, neighborhood_overview, neighbourhood, neighbourhood_group_cleansed

## 4. Cột nhóm định dùng bị rỗng / lỗi nhiều
- `price` (calendar): 2025-09-14 [FAIL, lỗi 100%], 2025-12-13 [ABSENT, lỗi 100%], 2026-03-21 [ABSENT, lỗi 100%], 2026-06-19 [ABSENT, lỗi 100%]
- `price` (listings): 2025-12-13 [FAIL, lỗi 100%]
  Hướng xử lý (nhóm chọn): bỏ cột / chỉ dùng ở mốc có dữ liệu / thay bằng cột khác (ví dụ bathrooms -> bathrooms_text). `price` theo đề chỉ dùng ở mốc cuối.
- Mức WARN (lỗi 5-30%): bathrooms@2025-09-14; bathrooms@2025-12-13; bathrooms@2026-06-19; bedrooms@2026-06-19; beds@2025-09-14; beds@2025-12-13; price@2025-09-14; review_scores_rating@2025-09-14; review_scores_value@2025-09-14

## Phạm vi ngày của calendar (để làm chuỗi theo mùa/tháng)
- Mốc 2025-09-14: 2025-09-14 -> 2026-09-17 (369 ngày)
- Mốc 2025-12-13: 2025-12-13 -> 2027-01-01 (385 ngày)
- Mốc 2026-03-21: 2026-03-21 -> 2027-04-01 (377 ngày)
- Mốc 2026-06-19: 2026-06-19 -> 2027-06-30 (377 ngày)

## 5. Gợi ý kết luận
- Mọi số đếm khớp kỳ vọng.
**Gợi ý: xem lại các mục trên; nếu lỗi nằm ở cột bắt buộc ở mốc không thay thế được thì báo giảng viên cấp lại.**