# Lab Day 10: tiến độ và ước lượng (14/10/2026)

File này là ảnh chụp tiến độ lúc E2 đang chạy. Chỉ `report/evidence/*.json` do `scripts/score.py` tạo sau khi chấm đủ 311 ảnh test mới là kết quả thí nghiệm chính thức.

## Đã đo được

| Mục | Số liệu | Nguồn |
|---|---:|---|
| E0 F1 | 0.0795 | `report/evidence/e0.json` |
| E1 F1 | 0.2806 | `report/evidence/e1.json` |
| E2 bước huấn luyện | 100/300 | `work/logs/remaining.log` lúc 11:43 ngày 14/10/2026 (giờ Việt Nam) |
| E2 loss bước 1 → 100 | 1.0070 → 0.6582 | Log huấn luyện; loss theo batch nên không giảm đều ở mọi bước |
| E2 tham số được cập nhật | 142,135,296 | `apply_setup` in lúc khởi chạy; nhiều hơn E1 109,105,152 tham số |
| GPU đang dùng | khoảng 3.8/4 GB | `nvidia-smi` khi E2 đang chạy; đây không phải `peak_vram_gb` cuối thí nghiệm |

Checkpoint E2 ở bước 100 nằm trong `work/e2/` và được Git bỏ qua theo yêu cầu bài lab. E2 chưa có F1 hoặc CI95 vì chưa suy luận và chấm tập test. E3 và E4 chưa bắt đầu.

## Ước lượng, không phải kết quả đo

E2 bắt đầu khoảng 05:07 và đạt bước 100 lúc 11:43 ngày 14/10/2026, tức khoảng 6 giờ 36 phút cho 100 bước đầu. Nếu tốc độ giữ nguyên, **200 bước huấn luyện còn lại mất khoảng 13 giờ 12 phút**, chưa gồm thời gian nạp model, suy luận 311 ảnh và chấm điểm. Đây là phép ngoại suy tuyến tính từ một mốc tiến độ; GPU 4 GB có thể chậm thêm hoặc hết bộ nhớ. Chưa đủ số liệu để dự đoán F1 của E2/E3/E4 một cách đáng tin cậy.

Tiến trình nền chạy lần lượt E2 → E3 → E4. `work/remaining_status.json` và `work/logs/remaining.log` cho biết trạng thái hiện tại. Sau khi đủ bằng chứng và `scripts/check_submission.py` đạt, `publish_when_complete.py` sẽ cập nhật báo cáo, commit và push `main` lên `origin`.
