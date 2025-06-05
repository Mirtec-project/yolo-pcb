from ultralytics import YOLO

dataset_path = "yolo_dataset2/data.yaml"

# 사용자 정의 모델 구조를 기반으로 YOLO 모델 생성
model = YOLO('yolo-pcb_0320.yaml')

# 모델 학습 진행
results = model.train(
    data=dataset_path,
    epochs=10000,
    imgsz=3904,
    device='cuda',
    batch=1,
    pretrained=False,
    patience=50,
    amp=True,                # 혼합 정밀도 학습 활성화
    fraction=0.75,            # GPU 메모리 사용량 제한
    cos_lr=True,             # 코사인 학습률 스케줄링
    cache=False,             # 캐시 비활성화
    close_mosaic=0,          # 모자이크 증강 비활성화
)

print(results)
