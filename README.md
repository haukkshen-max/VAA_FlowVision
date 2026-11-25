# VAA FlowVision — 1 Camera / Zone Analytics

**Phiên bản chỉnh từ V6 (đa camera) theo hướng chỉ dùng một camera cố định để phân tích đông người qua video.** Giao diện chính KHÔNG có phối hợp đa camera, không dùng dữ liệu thử nghiệm WILDTRACK làm số liệu của video thực.

## Bài toán mới

1. Đặt **một camera cố định** ở một khu vực, ví dụ sảnh ra vào / hành lang / khu chờ.
2. Chọn video có sẵn (ví dụ video dài 30 phút). Không cần stream camera thực tế để demo.
3. Đánh dấu **2–4 khu vực** trên cùng một khung ảnh: chia ngang, chia dọc hoặc hình chữ nhật tùy chỉnh. Tên khu có thể đổi (VD: Quầy, Ghế chờ, Lối đi).
4. YOLO11n (đóng gói trong thư mục `assets/model`) phát hiện người; mỗi người được gán vùng theo **điểm giữa cạnh đáy của bounding box**.
5. Phân tích số người từng khung hình, trung bình/đỉnh từng khu, mức vượt ngưỡng tham khảo, phút nào đông, biểu đồ nhiệt theo phút.
6. Xuất **ZIP gồm báo cáo HTML offline, CSV theo khung/từng phút/từng khu, JSON, ảnh minh chứng**, tùy chọn video đánh dấu.

## Cài và chạy trên Windows

- Cài Python 3.10 trở lên, nên dùng Python 3.11. Cấu hình GPU CUDA nếu có, CPU vẫn chạy nhưng video 30 phút có thể mất nhiều thời gian.
- Giải nén ZIP, mở thư mục chứa `app.py`.
- Nhấp đúp **`run_app.bat`**. Lần đầu hệ thống sẽ cài gói Python (cần internet).
- Mở `http://localhost:8501`, chọn video, phân vùng, nhấn **Bắt đầu phân tích video**.

Hoặc trong Terminal:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS/Linux: `./run_app.sh`.

## Cách demo video 30 phút

Ví dụ, nguồn video dài 30 phút, FPS gốc 25, cấu hình lấy mẫu **1 FPS**: hệ thống phân tích khoảng **1.800 khung hình** trải đều từ đầu tới cuối đoạn 30 phút. Đây là *số khung hình được lấy mẫu*, KHÔNG phải 1.800 người. Mỗi khung được đếm người ở khu A, B, C; báo cáo tổng hợp theo phút và theo khu. Số thực tế chỉ có sau khi chạy video.

- Nếu CPU chậm: chọn 0,5 FPS và độ phân giải 480/640.
- Nếu người xuất hiện rất nhỏ: thử 960/1280, nếu máy đủ mạnh.
- Với video >1 GB, khuyến nghị chọn **Đường dẫn video trên máy** thay vì tải lên giao diện để tránh tràn RAM. Ở chế độ đường dẫn, video phải nằm trên **máy chạy Streamlit**.
- Chọn ROI nếu chỉ muốn AI xử lý một phần ảnh; khu A/B/C vẫn tính theo **tọa độ toàn khung**.
- Mặc định không cần Track ID, đếm số người hiện diện trong các khung hình.
- Nếu đánh dấu video, MP4 đầu ra chỉ chứa các khung hình đã phân tích, phát tăng tốc để demo, không phải bản video nguyên gốc 30 phút. Nếu máy có `ffmpeg` trong PATH, hệ thống sẽ tự chuyển sang H.264 để trình duyệt dễ phát hơn; nếu không có, MP4V có thể chỉ xem bằng VLC.

## Định nghĩa số liệu (rất quan trọng)

| Chỉ số | Ý nghĩa |
| --- | --- |
| `total` | Số hộp người YOLO thấy trong một khung hình lấy mẫu |
| `mean_people_per_frame` | Trung bình tổng người / khung mẫu trên khoảng phân tích |
| `peak_people_in_frame` | Số người phát hiện nhiều nhất **tại cùng một khung hình** |
| `Khu A`/`B`/`C` | Số người có điểm chân nằm trong vùng ảnh tương ứng tại thời điểm đó |
| `mean_people` | Trung bình người / khung ở một khu; chọn giá trị cao nhất làm **khu đông nhất** |
| `total_detection_events` | Tổng lần phát hiện cộng dồn qua các khung. **Một người có thể bị đếm lặp nhiều lần.** |
| `estimated_track_ids` | Tùy chọn thử nghiệm, số Track ID trong phiên. ID có thể phân mảnh khi bị che khuất, không coi là lượt khách/người duy nhất đã được xác thực. |
| `estimated_over_capacity_minutes` | Thời gian vượt **ngưỡng do người dùng nhập**, suy từ các khung lấy mẫu; không phải đánh giá an toàn chính thức. |
| `outside_zones` | Người phát hiện trong khung/ROI nhưng điểm chân không nằm trong khu đã định nghĩa. |

**Không thể suy ra mật độ người/m² chỉ từ ba vùng ảnh** khi chưa có phép hiệu chuẩn thực địa. Các vùng tùy chỉnh nên KHÔNG chồng nhau; nếu chồng, một người chỉ được gán vùng khai báo trước. Một camera có vùng bị khuất, góc chết, che người nhau, nên tổng số người trong khu vực thực tế có thể cao hơn số phát hiện.

**Không có số liệu demo nào được dựng sẵn rồi gọi là thật.** Giao diện chỉ hiển thị số liệu sau khi xử lý video người dùng cung cấp.

## Phân tích video dài từ lệnh CMD (không phụ thuộc giao diện)

```bash
python analyze_cli.py --video "D:\Videos\camera.mp4" --duration-min 30 --sample-fps 1 --imgsz 640
```

Tùy chọn `--zones-json zones_example.json`, `--orientation horizontal`, `--device cpu`, `--output duong_dan_ket_qua`. `analyze_cli.py --help` để xem đủ.

## Bảo mật và lưu dữ liệu

App không nhận diện khuôn mặt hay danh tính. File video tải lên giao diện được lưu tạm **trên máy đang chạy Streamlit** trong `data/runtime`, không tự gửi bên thứ ba ngoại trừ quy trình tải thư viện/model khi cài đặt. Kết quả CSV/báo cáo lưu trong `data/runtime/session_*`. Không chia sẻ video người khác nếu không có quyền; xóa `data/runtime/upload_*` và `session_*` khi hết nhu cầu. Video gốc không đóng gói chung báo cáo ZIP.

## Thành phần

- `app.py`: giao diện Streamlit 1 camera / chọn video / phân vùng / báo cáo.
- `src/fixed_camera.py`: phân tích video, đếm người, thống kê, ảnh và báo cáo.
- `assets/model/yolo11n.pt`: model YOLO11n pretrained đóng gói sẵn từ V6.
- `analyze_cli.py`: giao diện dòng lệnh để xử lý video dài.
- `tests/test_fixed_camera.py`: bộ test logic với video giả lập và model giả lập, không thay thế kiểm tra độ chính xác YOLO ngoài đời.

## Kiểm thử

```bash
python -m pip install pytest
python -m pytest -q tests/test_fixed_camera.py
```

Các test kiểm tra gán vùng, tính đếm/đỉnh/thống kê, ROI, timestamp, xuất báo cáo. **Chưa có video thực 30 phút được người dùng cung cấp** nên gói này chưa có kiểm định sai số trên địa điểm thực tế; trước khi dùng nghiên cứu/kết luận nên gán nhãn tay một số khung ngẫu nhiên, tính MAE/Precision/Recall tại địa điểm.

> Lưu ý triển khai: đây là **ứng dụng Python Streamlit** chạy trên máy/chủ Python, không phải website `index.html` chạy trực tiếp trên Cloudflare Pages tĩnh.
