"""# Hướng dẫn gán nhãn tay

Hai người gán ĐỘC LẬP; chỗ bất đồng thì thảo luận và sửa hướng dẫn này, rồi gán lại.

- sentiment: positive = chủ yếu khen; negative = chủ yếu chê; mixed = có cả khen lẫn chê rõ ràng, hoặc không nghiêng hẳn về bên nào. Review không có nội dung: mixed.
- aspects_positive / aspects_negative: chỉ ghi khía cạnh THỰC SỰ được nhắc tới. Danh mục cố định:
  location (cả khu phố, giao thông), cleanliness, host, value, noise, amenities (wifi, giường, bếp, phòng tắm...), safety.
- Mỗi khía cạnh chỉ thuộc MỘT danh sách, theo ý chính của review (không ghi cùng một khía cạnh ở cả hai).
- noise: yên tĩnh = khen noise; ồn ào = chê noise. safety: thấy an toàn = khen; thấy không an toàn = chê.
- Nhận xét chung về chỗ ở (không nêu cụ thể) thì KHÔNG gán amenities.
- language: mã ISO 639-1 của ngôn ngữ chính (en, es, fr...); "und" nếu chỉ có emoji hoặc không xác định được.
- Các danh sách nối bằng dấu | ; để trống nếu không có.
"""