" YOLOV8_training_lung_cancer_detect" 

I Giới thiệu

Dự án này xây dựng và huấn luyện mô hình **YOLOv8** cho bài toán **phát hiện khối u phổi (lung cancer detection)** trên ảnh y tế (CT/X-ray).  
Mục tiêu là áp dụng **Object Detection** để khoanh vùng các vùng nghi ngờ ung thư phổi, hỗ trợ sàng lọc và chẩn đoán ban đầu.

II Mục tiêu
- Huấn luyện mô hình YOLOv8 cho bài toán phát hiện khối u phổi
- Đánh giá hiệu năng mô hình bằng các chỉ số chuẩn trong Computer Vision
- Phân tích kết quả dự đoán thông qua Confusion Matrix và trực quan hóa
- 
III Mô hình và thư viện được áp dụng .
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy, Matplotlib
- Google colab Notebook

YOLOv8 là mô hình object detection một giai đoạn (one-stage detector) với ưu điểm tốc độ nhanh, độ chính xác cao và phù hợp cho các bài toán triển khai thực tế.

IV Dữ liệu
- Nguồn dữ liệu được lấy từ Kaggle có lên là: Lung Nodules Detection Dataset Annotations
- Dữ liệu ảnh y tế phổi (CT)
- Định dạng nhãn: YOLO
- Chia tập dữ liệu:
  - Training
  - Validation

Các lớp (Classes)
- `class_0`: Không có khối u
- `class_1`: Có khối u
- `background`: Nền


V. Cấu hình huấn luyện

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

- Framework: Ultralytics YOLOv8 (PyTorch)

VI. Kết quả training 

+ Loss
- Training loss giảm đều theo epoch 
- Validation loss ổn định, không có dấu hiệu overfitting rõ rệt  
→ Hai điều này chứng tỏ mô hình hội tụ tốt và học ổn định.



+ Precision & Recall
- Precision ≈ 0.80 – 0.85
- Recall ≈ 0.90 – 0.95

→ Mô hình phát hiện đúng khối u với độ nhạy cao, hạn chế false negative – yếu tố quan trọng trong bài toán y tế.

+ mAP (Mean Average Precision)
- mAP@0.5 ≈ 0.85
- mAP@0.5:0.95 ≈ 0.45

→ Mô hình có khả năng định vị đối tượng tốt ở nhiều ngưỡng IoU khác nhau.

+ Confusion Matrix

| Predicted \\ True | class_0 | class_1 | background |
|------------------|--------|--------|------------|
| **class_0** | 31 | 0 | 2 |
| **class_1** | 1 | 8 | 4 |
| **background** | 4 | 0 | 0 |

+ Nhận xét: 
- Phân biệt tốt giữa `class_0` và `class_1`.
- Một số mẫu background vẫn bị nhầm với class_1.
- Số lượng false negative thấp, phù hợp với bài toán phát hiện bệnh.

+ Trực quan hóa
- Kết quả dự đoán được hiển thị trực tiếp trên ảnh với bounding box
- Mô hình khoanh vùng tương đối chính xác các khu vực nghi ngờ khối u


+ Hạn chế
- Dataset còn tương đối nhỏ
- Một số nhầm lẫn với background
- Chưa đánh giá trên nhiều nguồn dữ liệu khác nhau

+ Hướng phát triển
- Mở rộng và đa dạng hóa dataset
- Fine-tune thêm hyperparameters
- Triển khai demo inference bằng Gradio hoặc Flask
- So sánh với các mô hình khác (CNN, Faster R-CNN)

VII. Kết luận
Dự án cho thấy YOLOv8 hoàn toàn khả thi cho bài toán phát hiện ung thư phổi trên ảnh y tế (CT).  
Kết quả đạt được chứng minh khả năng áp dụng Deep Learning và Computer Vision vào các bài toán y sinh thực tế.
Dự án được thực hiện với mục đích học tập và nghiên cứu, không thể thay thế chẩn đoán y khoa chuyên nghiệp.
