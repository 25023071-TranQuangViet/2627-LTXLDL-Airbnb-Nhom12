Công cụ đã dùng: Claude Sonet 5.5
Công cụ được sử dụng vào việc:
    + Hỗ trợ sinh ra code gợi ý từ những ý tưởng, hướng xử lý của nhóm đã đề xuất ra
    + Hỗ trợ đề xuất ra nhu cầu thông tin của đối tượng nhóm hướng tới là khách du lịch (ví dụ khách du lịch cần thông tin gì về giá, độ sẵn sàng,...) để từ đó có hướng để tính toán kpi
    + Hỗ trợ các thành viên trong nhóm làm quen và xử lý các lỗi thường gặp khi làm việc nhóm trên github

Các prompt then chốt:
    + "Hãy gợi ý giúp tôi code phần qa_listings.py để làm sạch các dữ liệu thô bằng các cách như sau: gắn cờ các listing_id trong bảng listing_loc mà không có trong bảng listings thô, kiểm tra xem trong bảng có các cột cần thiết được liệt kê sẵn trong file qa tôi gửi không, xử lý kiểu dữ liệu các cột như price, bathroom_text, last_scraped,... , và gán cờ các dữ liệu bị khuyết thiếu, dữ liệu ngoại lai và dữ liệu bị trùng lặp"
    
