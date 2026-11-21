Công cụ đã dùng: Claude Sonet 5.5, Gemini Flash
Công cụ được sử dụng vào việc:
    + Hỗ trợ sinh ra code gợi ý từ những ý tưởng, hướng xử lý của nhóm đã đề xuất ra
    + Hỗ trợ các thành viên trong nhóm làm quen và xử lý các lỗi thường gặp khi làm việc nhóm trên github
    + Hỗ trợ gợi ý đưa ra dàn ý, cấu trúc để làm báo cáo

Các prompt then chốt:
    + "Hãy gợi ý giúp tôi code phần qa_listings.py để làm sạch các dữ liệu thô bằng các cách như sau: gắn cờ các listing_id trong bảng listing_loc mà không có trong bảng listings thô, kiểm tra xem trong bảng có các cột cần thiết được liệt kê sẵn trong file qa tôi gửi không, xử lý kiểu dữ liệu các cột như price, bathroom_text, last_scraped,... , và gán cờ các dữ liệu bị khuyết thiếu, dữ liệu ngoại lai và dữ liệu bị trùng lặp"
    + "Gợi ý giúp tôi yếu tố cần thêm vào phần gắn cờ trong calendar bên cạnh gắn cờ null, ngày đêm vượt ngưỡng,.. và chỉ cách xuất ra report ra data/processed"
    + " Viết code sinh hình cho các kpi theo như sự thống nhất dạng biểu đồ đã chọn sử dụng thư viện matplotlib, yêu cầu tạo ảnh định dạng pdf"
    + "Hãy gợi ý giúp tôi code phần kpi để tính giá trung vị trên đầu người của từng quận và so sánh giá trung vị đó với mặt bằng chung của toàn thành phố, làm tương tự với điểm đánh giá trung bình, tính tỷ lệ còn trống theo từng mốc, từng tháng và tỷ lệ phàn nàn về tiếng ồn, an toàn"
    + "Gợi ý giúp tôi cấu trúc phần baseline"

Các lỗi của AI và cách nhóm sửa
    + Tại phần qa_listings, AI chỉ gắn cờ các giá trị thiếu nhưng không điền khuyết. Nhóm đã xử lý điền khuyết dữ liệu bằng cách: với cột price_clean, nhóm đã gộp thành các nhóm theo loại phòng và sức chứa của phòng rồi tính và gán trung vị của nhóm bằng transform, với nhóm mà không có giá nào thì điền bằng giá trung vị của cả cột price_clean, với các cột bedrooms, beds và bathrooms cũng làm tương tự
    + Tại phần qa_reviews, AI chỉ xử lý các dòng bị trùng lặp review_id trong cùng 1 mốc nhưng khi concat bảng review của 4 mốc lại với nhau thì lại xảy ra hiện tượng các review_id và date từ các mốc khác nhau lại bị trùng. Nhóm đã gán thêm cờ trùng lặp giữa các mốc vào để khi dùng trong llm sẽ bỏ phần trùng lặp đó đi
    + Tại 
