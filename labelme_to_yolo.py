#!/usr/bin/env python
"""
LabelMe 형식의 dataset (이미지와 JSON 파일)을 YOLO (Ultralytics) 형식으로 변환하는 스크립트입니다.

JSON 파일의 "shapes" 목록에서
- "rectangle"인 경우 – 두 점을 이용해 4개의 corner (좌상, 우상, 우하, 좌하) 좌표로 변환한 후 정규화합니다.
- "polygon"인 경우 – 이미 4점의 좌표가 있다고 가정하고 그대로 사용합니다.

데이터셋은 train, val, test (8:1:1 비율)로 나누어지고 각 디렉토리 하위에 images와 labels 폴더가 생성됩니다.
또한, classes와 데이터 경로를 포함한 data.yaml 파일도 생성합니다.
"""

import os
import json
import glob
import random
import shutil
import argparse
from pathlib import Path
import yaml  # PyYAML 설치 필요

def convert_rectangle_to_polygon(points):
    """
    LabelMe의 rectangle annotation은 두 개의 좌표로 제공됩니다.
    이를 (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax) 4점으로 변환합니다.
    """
    x1, y1 = points[0]
    x2, y2 = points[1]
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

def process_json_file(json_path, class_mapping):
    """
    json 파일을 파싱하여 이미지 파일명과 YOLO annotation 라인(정규화된 좌표 기반 4개 점)을 반환합니다.
    """
    with open(json_path, 'r') as f:
         data = json.load(f)
    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    image_file = data.get("imagePath")
    if not image_file:
         raise ValueError(f"JSON 파일 {json_path}에 imagePath가 없습니다.")
    annotations = []
    for shape in data.get("shapes", []):
         label = shape.get("label")
         if label not in class_mapping:
              continue
         class_id = class_mapping[label]
         shape_type = shape.get("shape_type", "polygon")
         points = shape.get("points")
         # points가 2개이면 rectangle으로 간주하여 4개 좌표로 변환
         if len(points) == 2 or shape_type == "rectangle":
              points = convert_rectangle_to_polygon(points)
         # polygon이면 4좌표가 있다고 가정합니다.
         if not points or len(points) != 4:
              print(f"경고: {json_path}의 shape에서 좌표 개수가 4가 아니어서 스킵합니다. points: {points}")
              continue
         # 각 좌표를 이미지 크기로 나누어 정규화합니다.
         norm_points = []
         for pt in points:
              norm_x = pt[0] / img_w
              norm_y = pt[1] / img_h
              norm_points.extend([norm_x, norm_y])
         # annotation 라인: 클래스번호와 8개의 좌표 값 (소수점 6자리까지)
         ann_line = f"{class_id} " + " ".join(f"{p:.6f}" for p in norm_points)
         annotations.append(ann_line)
    return image_file, annotations

def create_dataset_structure(output_dir):
    """
    output_dir 하위에 train, val, test 디렉토리 및 그 안의 images와 labels 폴더를 생성합니다.
    """
    splits = ['train', 'val', 'test']
    for split in splits:
         for sub in ['images', 'labels']:
              dir_path = os.path.join(output_dir, split, sub)
              os.makedirs(dir_path, exist_ok=True)

def generate_yaml_file(output_dir, class_list):
    """
    YOLO 학습에 사용될 data.yaml 파일을 생성합니다.
    """
    yaml_dict = {
         "train": os.path.abspath(os.path.join(output_dir, "train", "images")),
         "val": os.path.abspath(os.path.join(output_dir, "val", "images")),
         "test": os.path.abspath(os.path.join(output_dir, "test", "images")),
         "nc": len(class_list),
         "names": class_list
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
         yaml.dump(yaml_dict, f, sort_keys=False)
    print("YAML 파일 저장됨:", yaml_path)

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    random.seed(42)
    
    # input_dir 내의 모든 JSON 파일을 찾습니다.
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not json_files:
         print("입력 디렉토리에서 JSON 파일을 찾을 수 없습니다:", input_dir)
         return
    
    # 고정된 클래스 설정: "CMounting"과 "CSolder"만 사용합니다.
    expected_classes = ["CMounting", "CSolder"]  # 반드시 이 순서대로 사용. (클래스 0 과 1에 해당)
    class_mapping = {label: idx for idx, label in enumerate(expected_classes)}
    class_list = expected_classes
    print("사용할 클래스:", class_list)
    
    # YOLO 데이터셋 디렉토리 구조 생성
    create_dataset_structure(output_dir)
    
    # JSON 파일 셔플 후 8:1:1 비율로 분할
    random.shuffle(json_files)
    n_total = len(json_files)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val   # 나머지를 test로 지정
    splits_dict = {
         'train': json_files[:n_train],
         'val': json_files[n_train:n_train+n_val],
         'test': json_files[n_train+n_val:]
    }
    
    # 각 분할별로 annotation 변환 및 이미지 복사 처리
    for split, files in splits_dict.items():
         print(f"{split} 세트 처리 중 (파일 수: {len(files)})...")
         for json_file in files:
              try:
                  image_file_name, ann_lines = process_json_file(json_file, class_mapping)
              except Exception as e:
                  print(f"{json_file} 처리 중 에러 발생: {e}")
                  continue
              # JSON 파일의 basename으로 label 파일 이름 생성 (.txt)
              base_name = os.path.splitext(os.path.basename(json_file))[0]
              label_output_path = os.path.join(output_dir, split, "labels", base_name + ".txt")
              with open(label_output_path, 'w') as f:
                   for line in ann_lines:
                        f.write(line + "\n")
              # JSON 내의 imagePath에 해당하는 이미지 파일 복사
              img_src = os.path.join(input_dir, image_file_name)
              if not os.path.exists(img_src):
                   # JSON 파일과 이미지가 같은 폴더에 있는 경우
                   img_src = os.path.join(os.path.dirname(json_file), image_file_name)
              if os.path.exists(img_src):
                   img_dst = os.path.join(output_dir, split, "images", image_file_name)
                   shutil.copy2(img_src, img_dst)
              else:
                   print(f"경고: {json_file}에 명시된 이미지 {image_file_name}을 찾을 수 없습니다.")
    
    # data.yaml 파일 생성
    generate_yaml_file(output_dir, class_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LabelMe 데이터셋을 YOLO (Ultralytics) 형식으로 변환합니다.")
    parser.add_argument("--input_dir", type=str, default=".", help="LabelMe JSON 파일과 이미지가 있는 입력 디렉토리")
    parser.add_argument("--output_dir", type=str, default="./yolo_dataset", help="YOLO 형식 데이터셋이 저장될 출력 디렉토리")
    args = parser.parse_args()
    main(args)
