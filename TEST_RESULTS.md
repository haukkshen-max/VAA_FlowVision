# Kết quả kiểm thử sản phẩm Single Camera V7

- **Đạt 8/8 test logic offline** bằng `pytest` trên Python hiện có.
- Có kiểm thử video tổng hợp với **timeline dài 30 phút** (1.800 khung, 1 FPS), dùng model giả (`FakeYOLO`). Xác minh video dài không bị giới hạn 15 giây như V6, có đủ 30 dòng thống kê theo phút và phân vùng đúng.
- Có kiểm thử video 4 giây với nhiều đối tượng thử: phân vùng A/B/C, peak/mean/count, ROI, track ID thử nghiệm, khu vực ngoài phạm vi, frame không có người, MP4 đã đánh dấu và xuất ZIP báo cáo.
- Xác minh file model `assets/model/yolo11n.pt` là archive PyTorch hợp lệ, CRC không lỗi.
- Biên dịch cú pháp toàn bộ `app.py`, `src` và CLI thành công.

## Chưa thể xác nhận trong môi trường kiểm thử

- **Suy luận YOLO11n thật và render giao diện Streamlit**: môi trường dựng gói không có `ultralytics` và `streamlit` và không thể cài từ PyPI do không có mạng. Máy dùng thực tế cài bằng `python -m pip install -r requirements.txt` (có internet).
- **Sai số trên video camera thật 30 phút**: người dùng chưa cung cấp video đó, không có số người/ngưỡng chính xác để công bố. Các số liệu sẽ được sinh từ video người dùng khi chạy app, không lấy số liệu WILDTRACK cũ hay số liệu giả làm kết quả thực.

Để kiểm tra logic tại máy của bạn: `python -m pip install pytest` sau đó `python -m pytest -q tests/test_fixed_camera.py`.
