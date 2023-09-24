import numpy as np
import cv2
from sklearn.cluster import KMeans
import colorspacious as cs
import os
import time

# 색상 클래스 정의 (HTML 색 기준)
color_classes = {
    '블랙': [0, 0, 0], 
    '그레이': [128, 128, 128], '실버': [192, 192, 192], '딤그레이': [105, 105, 105],
    '그린': [0, 128, 0], '다크그린': [0, 100, 0], '옐로우그린': [154, 205, 50],
    '네이비': [0, 0, 128], '다크블루': [0, 0, 139], '미드나잇블루': [25, 25, 112],
    '라벤더': [230, 230, 250], 
    '레드': [255, 0, 0],  '인디안레드': [205, 92, 92],
    '민트': [201, 236, 216],  # 민트 - HTML 색에 없음
    '베이지': [245, 245, 220], '아이보리': [255, 255, 240],
    '브라운': [165, 42, 42], '새들브라운': [139, 69, 19],
    '블루': [0, 0, 255], '스틸블루': [70, 130, 180],
    '스카이블루': [135, 206, 235], '딥스카이블루': [0, 191, 255], '아쿠아': [0, 255, 255],
    '옐로우': [255, 255, 0], '골드': [255, 215, 0], '레몬쉬폰': [255, 250, 205],
    '오렌지': [255, 165, 0], '다크오렌지': [255, 140, 0], '코랄': [255, 127, 80],
    '와인': [114, 47, 55],  
    '카키': [189, 183, 107], 
    '퍼플': [128, 0, 128], '미디엄퍼플': [147, 112, 219],
    '핑크': [255, 192, 203], '핫핑크': [255, 105, 180], '딥핑크': [255, 20, 147],
    '화이트': [255, 255, 255], '스노우': [255, 250, 250], '화이트스모크': [245, 245, 245]
}
class_mapping = {
    '블랙': '블랙', '그레이': '그레이', '실버': '그레이', '딤그레이': '그레이', '그린': '그린', '다크그린': '그린', '옐로우그린': '그린',
    '네이비': '네이비', '다크블루': '네이비', '미드나잇블루': '네이비', '라벤더': '라벤더', '레드': '레드', '인디안레드': '레드',
    '민트': '민트', '베이지': '베이지', '아이보리': '베이지', '브라운': '브라운', '새들브라운': '브라운', '블루': '블루', '스틸블루': '블루',
    '스카이블루': '스카이블루', '딥스카이블루': '스카이블루', '아쿠아': '스카이블루', '옐로우': '옐로우', '골드': '옐로우', '레몬쉬폰': '옐로우',
    '오렌지': '오렌지', '다크오렌지': '오렌지', '코랄': '오렌지', '와인': '와인', '카키': '카키', '퍼플': '퍼플', '미디엄퍼플': '퍼플',
    '핑크': '핑크', '핫핑크': '핑크', '딥핑크': '핑크', '화이트': '화이트', '스노우': '화이트', '화이트스모크': '화이트'
}

def extract_dominant_color(image_path, k=5):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    assert image.shape[2] == 4, "이미지는 알파 채널을 포함해야 합니다."

    # RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    # 알파 채널을 사용하여 배경이 아닌 픽셀만 선택하고 알파 채널 제거
    mask = image[:, :, 3] > 0
    image_rgb = image[mask, :3]

    if image_rgb.shape[0] > 500_000:
        step_size = 5  # 리샘플링 간격
        image_rgb = image_rgb[::step_size]

    kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++')
    kmeans.fit(image_rgb)
    dominant_colors = kmeans.cluster_centers_

    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    proportions = counts / len(image_rgb)

    # proportions 기준으로 내림차순 정렬
    sorted_indices = np.argsort(proportions)[::-1]
    dominant_colors = dominant_colors[sorted_indices]
    proportions = proportions[sorted_indices]

    # 클러스터링 결과에서 각 클러스터의 중심을 반환
    return dominant_colors, proportions


def closest_color_class(dominant_color):
    # 각 색상 클래스를 Lab 공간으로 변환
    color_keys = list(color_classes.keys())
    lab_colors = np.array([cs.cspace_convert(color_classes[color], "sRGB255", "CIELab") for color in color_keys])

    # 주요 색상을 Lab 공간으로 변환
    dominant_lab = cs.cspace_convert(dominant_color, "sRGB255", "CIELab")

    # 배열 연산으로 각 색상 클래스와 주요 색상 간의 거리 계산
    distances = np.array(
        [cs.deltaE(dominant_lab, lab, input_space="CIELab", uniform_space="JCh") for lab in lab_colors])

    # 가장 가까운 색상 클래스 찾기
    closest_color_key = color_keys[np.argmin(distances)]

    return class_mapping[closest_color_key]

def aggregate_colors(dominant_colors, proportions):
    color_class_mapping = {}

    for color, proportion in zip(dominant_colors, proportions):
        color_class = closest_color_class(color)

        if color_class in color_class_mapping:
            color_class_mapping[color_class] += proportion
        else:
            color_class_mapping[color_class] = proportion

    # Sort by proportions in descending order
    sorted_colors = sorted(color_class_mapping.items(), key=lambda x: x[1], reverse=True)

    return sorted_colors

if __name__ == "__main__":
    directory_path = "C:\\Users\\SSAFY\\Downloads\\test_dataset"

    start_time = time.time()
    # 하나의 사진에 대한 추론
    file_name = "test_30.png"
    image_path = os.path.join(directory_path, file_name)
    dominant_colors, proportions = extract_dominant_color(image_path)

    print(image_path)
    sorted_colors = aggregate_colors(dominant_colors, proportions)

    # 백분율로 변환하고, 문자열로 형식화
    formatted_colors = [f"'{color}': {percentage * 100:.2f}%" for color, percentage in sorted_colors]
    # 결과 문자열 생성
    result_str = ', '.join(formatted_colors)
    print(result_str)

    predicted = sorted_colors[0][0]
    if(len(sorted_colors) >= 2 and (sorted_colors[0][1] < 0.5 or sorted_colors[1][1] > 0.3)):
        if((sorted_colors[0][0] in ('네이비', '블루', '스카이블루') and sorted_colors[1][0] in ('네이비', '블루', '스카이블루'))
            or (sorted_colors[0][0] in ('핑크', '퍼플') and sorted_colors[1][0] in ('핑크', '퍼플'))
            or (sorted_colors[0][0] in ('그린', '민트') and sorted_colors[1][0] in ('그린', '민트'))
            or (sorted_colors[0][0] in ('베이지', '카키', '브라운') and sorted_colors[1][0] in ('베이지', '카키', '브라운'))):
            predicted = sorted_colors[0][0]
        else:
            predicted = '다채색'
    print(predicted)
    print(time.time() - start_time, 'seconds')
