from ultralytics import YOLO

def main():
    # 1. Khai báo đường dẫn đến file data.yaml (Sửa lại đường dẫn này cho đúng với máy bạn nhé)
    # Chữ 'r' ở đầu giúp Windows không bị lỗi nhận diện dấu gạch chéo ngược (\)
    duong_dan_data = r"D:/UEH/AI-TRAIN/3dfail/data.yaml"

    # 2. Khởi tạo mô hình YOLOv8 Nano
    print("Đang khởi tạo mô hình...")
    model = YOLO("yolov8n.pt")

    # 3. Tiến hành huấn luyện
    print("Bắt đầu huấn luyện trên RTX 3050...")
    results = model.train(
        data=duong_dan_data,
        epochs=50,
        imgsz=640,
        device=0,      # Bắt buộc là 0 để nhận Card NVIDIA
        batch=4,       # Giữ batch=4 để không bị tràn 4GB VRAM
        workers=2,     # Chống nghẽn CPU trên Windows
        name="nhan_dien_in_3d" 
    )
    
    print("Hoàn thành! Trọng số mô hình nằm trong thư mục: runs/detect/nhan_dien_in_3d/weights/")

# Bắt buộc phải có hàm này khi train YOLO trên Windows
if __name__ == '__main__':
    main()