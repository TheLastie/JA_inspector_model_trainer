from ultralytics import YOLO
import time
import os, cv2

from pathlib import Path

# Load the model
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
model = YOLO('yolo26x-pose.pt').to('cuda')


def main_image(file_path, input_dir, output_dir):
    """Обработка одного изображения: детекция позы и сохранение результата."""
    image_path = os.path.join(input_dir, file_path)
    results = model(image_path)
    output_path = os.path.join(output_dir, file_path)
    result = results[0]
    img_with_boxes = result.plot()
    keypoints = result.keypoints
    if keypoints is not None and len(keypoints) > 0:
        xy = keypoints.xy.cpu().numpy()
        conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
        for i in range(len(xy)):
            print(f"Человек {i}:")
            for j in range(17):
               x, y = xy[i, j]
               c = conf[i, j] if conf is not None else 1.0
               print(f"  Точка {j}: ({x:.1f}, {y:.1f}), уверенность {c:.2f}")
    else:
        print("Людей не обнаружено")

    height, width = img_with_boxes.shape[:2]
    new_width = 800
    new_height = int(height / width * new_width)

    resized = cv2.resize(img_with_boxes, (new_width, new_height), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized)


def main_video(file_path):
    """Обработка видео: извлечение ключевых точек покадрово."""
    st_time = time.time()

    cap = cv2.VideoCapture(file_path)
    all_dots = {}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = 600
    new_height = int(height / width * new_width)

    fr = 0
    while cap.isOpened():
        tm = time.time()
        fr += 1
        all_dots[fr] = {}
        ret, frame = cap.read()
        if not ret:
            del all_dots[fr]
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        results = model(resized_frame)[0]

        # Выбираем самую крупную фигуру по площади bounding box
        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        main_idx = areas.argmax()

        keypoints = results.keypoints
        if keypoints is not None and len(keypoints) > main_idx:
            xy = keypoints.xy.cpu().numpy()
            conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None

            for j in range(1, 14):
                x, y = xy[main_idx, j]
                c = conf[main_idx, j] if conf is not None else 1.0
                all_dots[fr].update({j: [(x, y), c]})

        print(f"Кадр {fr}: {time.time()-tm:.3f}с")

    cap.release()
    print(f"Итого: {time.time()-st_time:.1f}с, {fr} кадров")
    return all_dots
