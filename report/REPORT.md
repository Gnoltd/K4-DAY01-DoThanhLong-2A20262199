# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** : Sep 11 2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
    - {
        "class_id": 468,
        "class_name": "cab",
        "rank": 1,
        "score": 0.510915,
        "taxonomy_name": "ImageNet-1K"
    }   
- Record này mô tả toàn ảnh như thế nào? 
    - Ảnh có id 468 nhận dạng traffic là cab có số điểm cao nhất đạt rank 1: 0.51 tham chiếu tuân theo hệ thống phân loại của ImageNet-1K  
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    - ImageNet-1K
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    - class_id dùng để tra cứu 
    - class_name dùng cho người đọc
    - nhận biết class_id thuộc bộ nhãn nào 
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    - Single-Label “Một lớp cho mỗi ảnh” và nếu có nhiều chủ thể thì multi-label hoặc cách xử lý ảnh nhiều chủ thể khác
- Vì sao model score không phải ground truth?
    - score chỉ là confidence của model dự đoán và đưa ra nhận định trên những gì nó học chứ không phải kết quả được con người gán nhãn. Ground truth phải do con người

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    - {"class_name": "bus",
    "score": 0.912558,
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96}
- Diễn giải vị trí box bằng lời: 
    - đơn vị pixel, bắt đầu ở (93.17, 187.95) cách mép trái ~93px, mép trên ~188px kết thúc ở (223.01, 320.91). Box rộng ~130px và cao ~133px
- So sánh số prediction ở hai threshold:
    - so sánh số lượng prediction ở ngưỡng threshold = 0.35 và threshold = 0.6 theo nguồn `visuals/detection_predictions.png`:
        - threshold 0.35 có 11 vật thể 
        - threshold 0.6 có 6 vật thể
    * Với score < threshold thì sẽ không được lựa chọn
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    - nâng threshold sẽ giảm độ bao phủ nhưng ít prediction score thấp để reviewer cần xem hơn và ngược lại
- Đề xuất một quy tắc box chặt:
    - box ôm sát vật thể ở cả 4 cạnh, không chừa margin, cắt một phần vật thể
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    - Vẽ box cho phần nhìn thấy được hoặc ước lượng vẽ toàn bộ vật thể và gán nhãn bị che/mờ ngưỡng phần trăm thay vì bỏ qua

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    - {
    "instance_id": "kitchen-002",
    "class_name": "bowl",
    "score": 0.735744,
    "polygon_point_count": 67,
    "polygon_xy": [
      [53.0, 344.0],
      [52.0, 345.0],
      [50.0, 345.0]
    ]
    }
- Polygon bổ sung chi tiết gì so với box? 
    - là danh sách các điểm `[x, y]` theo pixel quanh biên instance.
- `instance_id` dùng để làm gì và không phải loại ID nào?
    - `instance_id` có dạng `<sample_id>-NNN`; đây là ID trong output lab, không phải class ID hoặc tracking ID.
- Đề xuất một quy tắc biên mask:
    - Đi sát biên ngoài cùng của pixel thuộc vật thể, không để hở và lấn sang vật thể khác
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    - khi vật thể bị che khuất một phần vẽ theo phần nhìn thấy hay ước lượng phần bị che, và khi hai vật thể dính sát nhau không rõ pixel nào thuộc bên nào thì phải escalate cho reviewer quyết định thay vì tự suy đoán, tránh làm sai lệch ground truth

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | `class_name`+`class_id`+`taxonomy` | ảnh có nhiều chủ thể (xe bus và ô tô) nhưng chỉ có một nhãn | gán một nhãn đại diện | nhãn có khớp với nội dung không |
| Phát hiện vật thể | `bbox_xyxy` + `class_name` | Vật thể bị che khuất/cắt mép ảnh |  Vẽ box ôm sát biên vật thể theo đúng quy tắc "box chặt"; nếu vật thể bị che/cắt mép thì vẽ phần lộ hoặc vẽ toàn bộ và đánh dấu che/mờ ngưỡng % | Box có chặt không; các vật thể cùng lớp có bị gắn thiếu box nào không |
| Instance segmentation | `polygon_xy` + `instance_id` | Hai instance cùng lớp dính sát nhau và ranh giới pixel giữa chúng không rõ ràng | Vẽ polygon bám sát biên pixel thực của từng instance, không chồng lấn sang instance/nền lân cận; gặp vùng dính/mờ thì escalate thay vì đoán | Biên polygon có sát viền thật không; `instance_id` có phân biệt đúng các object cùng lớp không; vùng biên mơ hồ đã được xử lý theo đúng escalation chưa |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: không cung cấp thông tin ra bên ngoài khi chưa có sự cho phép từ quản lý, không cung cấp dữ liệu cá nhân,...
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Báo phụ trách hoặc mentor và đối chiếu theo "xử lí lỗi thường gặp"

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
