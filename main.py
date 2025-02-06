from ultralytics import YOLO

dataset_path = "yolo_dataset/data.yaml"

# 사용자 정의 모델 구조를 기반으로 YOLO 모델 생성
model = YOLO('yolo-pcb_0206_1.yaml')

# 모델 학습 진행
results = model.train(
    data=dataset_path,
    epochs=10000,
    imgsz=1952,
    batch=1,
    pretrained=False
)

print(results)
