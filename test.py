from ultralytics import YOLO

# 사전 학습된 best.pt 파일을 이용하여 모델 로드
model = YOLO('best.pt')

# 모델 파라미터 수 계산 및 출력
try:
    if isinstance(model.model, list):
        # 앙상블 모델의 경우 각 모델의 파라미터 합산
        num_parameters = sum(sum(p.numel() for p in m.parameters()) for m in model.model)
    else:
        num_parameters = sum(p.numel() for p in model.model.parameters())
    print("모델 파라미터 수:", num_parameters)
except Exception as e:
    print("모델 파라미터 수를 확인할 수 없습니다:", e)

# 테스트 이미지들이 저장된 폴더 경로 (예: "test_images" 폴더)
test_images_folder = 'yolo_dataset2/test/images'  # 실제 폴더 경로로 변경하세요

# 모델 추론 실행 (박스만 표시하도록 라벨 및 신뢰도 제거 옵션 추가)
results = model.predict(source=test_images_folder, conf=0.5, hide_labels=True, hide_conf=True, save=True)

# 추론 결과 출력
for result in results:
    print(result)