# YOLOv8 Training for Lung Cancer Detection

## I. Giới thiệu

Dự án **YOLOv8 Training for Lung Cancer Detection** tập trung xây dựng và huấn luyện mô hình **YOLOv8** cho bài toán **phát hiện khối u phổi (lung cancer / lung nodule detection)** trên ảnh y tế (CT scan).

Mục tiêu của dự án là ứng dụng **Object Detection** để **khoanh vùng các vùng nghi ngờ khối u phổi**, từ đó hỗ trợ quá trình **sàng lọc và chẩn đoán ban đầu** trong lĩnh vực y sinh.

**Lưu ý**: Dự án mang tính học tập và nghiên cứu, **không thay thế cho chẩn đoán y khoa chuyên nghiệp**.

---

## II. Mục tiêu dự án

- Huấn luyện mô hình **YOLOv8** cho bài toán phát hiện khối u phổi
- Đánh giá hiệu năng mô hình bằng các chỉ số chuẩn trong Computer Vision
- Phân tích kết quả dự đoán thông qua **Confusion Matrix**
- Trực quan hóa kết quả detection trên ảnh y tế
- Đánh giá tính khả thi của YOLOv8 trong bài toán y sinh

---

## III. Mô hình và công nghệ sử dụng

### Mô hình & Framework
- **YOLOv8 (Ultralytics)**
- **PyTorch**

### Thư viện hỗ trợ
- OpenCV
- NumPy
- Matplotlib

### Môi trường huấn luyện
- Google Colab Notebook

YOLOv8 là mô hình **one-stage object detector**, nổi bật với:
- Tốc độ suy luận nhanh  
- Độ chính xác cao  
- Phù hợp cho các bài toán triển khai thực tế và nghiên cứu

---

## IV. Dữ liệu

- **Nguồn dữ liệu**: Kaggle  
  *Lung Nodules Detection Dataset Annotations*
- **Loại dữ liệu**: Ảnh CT phổi
- **Định dạng nhãn**: YOLO format
- **Chia tập dữ liệu**:
  - Training set
  - Validation set

### Các lớp (Classes)
- `class_0`: Không có khối u  
- `class_1`: Có khối u  
- `background`: Nền  

---

## V. Cấu hình huấn luyện

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    workers=2
)
```

## VI. Kết quả thực nghiệm (Experimental Results)

### 1. Quá trình huấn luyện

- Mô hình **YOLOv8** được huấn luyện ổn định, loss giảm đều theo các epoch
- Không xuất hiện hiện tượng **divergence** hay **overfitting nghiêm trọng**
- Validation loss duy trì xu hướng song song với training loss

Điều này cho thấy mô hình **học được đặc trưng dữ liệu hiệu quả**.

### 2. Chỉ số đánh giá (Evaluation Metrics)

| Metric | Giá trị |
|------|--------|
| Precision | ~0.82 |
| Recall | ~0.93 |
| mAP@0.5 | ~0.85 |
| mAP@0.5:0.95 | ~0.45 |

**Nhận xét**:
- Recall cao giúp **giảm false negative**, rất quan trọng trong bài toán phát hiện bệnh
- Precision ở mức tốt, hạn chế false positive
- mAP@0.5 đạt giá trị cao, cho thấy khả năng phát hiện ổn định ở ngưỡng IoU phổ biến

### 3. Confusion Matrix

| Predicted \\ True | class_0 | class_1 | background |
|------------------|--------:|--------:|-----------:|
| **class_0**      | 31      | 0       | 2          |
| **class_1**      | 1       | 8       | 4          |
| **background**   | 4       | 0       | 0          |

**Phân tích**:
- Mô hình phân biệt tốt giữa `class_0` và `class_1`
- Một số vùng nền (`background`) bị nhầm lẫn thành khối u
- Tỷ lệ **false negative thấp**, phù hợp cho bài toán y tế

### 4. Kết quả trực quan (Qualitative Results)

- Bounding box dự đoán bám sát vùng nghi ngờ khối u phổi
- Hoạt động tốt trên nhiều kích thước khối u khác nhau
- Dự đoán ổn định trên ảnh CT có độ nhiễu vừa phải

<p align="center">
  <img src="assets/sample_result.png" width="600"/>
</p>

> *Hình minh họa: Kết quả phát hiện khối u phổi bằng YOLOv8*

---

### 5. Đánh giá tổng quan

- Mô hình đạt hiệu năng **phù hợp cho nghiên cứu và thử nghiệm**
- Có tiềm năng ứng dụng trong hệ thống hỗ trợ chẩn đoán
- Cần thêm dữ liệu và fine-tuning để sử dụng trong môi trường thực tế
