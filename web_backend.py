#!/usr/bin/env python3
"""
Веб-сервер для анализа упражнений по загруженному видео.
Использует ансамбль моделей и возвращает аннотированное видео + JSON с сегментами.
"""

import os
import json
import tempfile
import shutil
import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid
import logging
import traceback
from typing import List, Optional, Dict, Any
from collections import deque

from model_zoo import RecurrentAutoencoder, TransformerAutoencoder, LSTMVAE
from experiment_manager import manager
from data_pipeline import load_all_frames
from joint_analysis_video import (
    load_ensemble, segment_video_simple,
    compute_joint_errors, compute_exercise_score, draw_skeleton_with_errors,
    process_video_with_yolo, find_reference_json, convert_score_to_5star,
    JOINT_NAMES, SKELETON_EDGES, ANGLE_TO_EDGES, ANGLE_TO_KEYPOINTS
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    raise RuntimeError("Ultralytics not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Exercise Analysis Service (Video Upload)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статическая папка для результатов
os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Загрузка моделей при старте
num_gpus = torch.cuda.device_count()
devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)] if num_gpus > 0 else [torch.device('cpu')]
primary_device = devices[0]
logger.info(f"Using primary device: {primary_device}, total GPUs: {num_gpus}")

ensemble_dir = "models/ensemble"
ensemble_models_dict, classifier, scaler, label_encoder = load_ensemble(ensemble_dir, primary_device, quiet=False)
logger.info(f"Loaded {len(ensemble_models_dict)} models")

# Распределение по GPU
ensemble_models = {}
model_to_device = {}
for i, (name, model) in enumerate(ensemble_models_dict.items()):
    dev = devices[i % len(devices)]
    model.to(dev)
    ensemble_models[name] = model
    model_to_device[name] = dev

yolo_model = YOLO("yolo26x-pose.pt").to(primary_device)
logger.info("YOLO loaded")

thresholds_cache = {}

def get_thresholds(exercise: str) -> np.ndarray:
    if exercise in thresholds_cache:
        return thresholds_cache[exercise]
    ref_json = find_reference_json(exercise)
    if ref_json and os.path.exists(ref_json):
        try:
            angles, _ = load_all_frames(ref_json)
            if len(angles) > 0:
                model = ensemble_models.get(exercise)
                if model:
                    dev = model_to_device.get(exercise, primary_device)
                    ref_norm = (angles / np.pi) * 2.0 - 1.0
                    ref_errors = compute_joint_errors(model, ref_norm, dev, seq_len=30)
                    thresh = np.percentile(ref_errors, 95.0, axis=0)
                    thresholds_cache[exercise] = thresh
                    return thresh
        except Exception as e:
            logger.error(f"Thresholds error for {exercise}: {e}")
    return np.full(11, 0.1, dtype=np.float32)

def process_uploaded_video(video_path: str, manual_exercise: Optional[str] = None):
    """
    Обрабатывает видео с помощью YOLO, сегментирует, вычисляет ошибки.
    Возвращает:
        frames (list of np.ndarray),
        annotated_frames (list of np.ndarray),
        segments (list of dict),
        overall_score (float)
    """
    ensemble_models = ensemble_models_dict
    logger.info(f"All {len(ensemble_models)} models remain on {primary_device}")
    # Захват углов и кадров
    frames, all_angles, all_keypoints, all_boxes = process_video_with_yolo(
        video_path, "yolo26x-pose.pt", primary_device, quiet=True
    )
    valid_indices = [i for i, a in enumerate(all_angles) if a is not None]
    if not valid_indices:
        raise ValueError("No pose detected in video")

    # Сегментация
    if manual_exercise:
        segments = [(valid_indices[0], valid_indices[-1], manual_exercise, 1.0)]
    else:
        segments = segment_video_simple(
            all_angles, valid_indices, ensemble_models, classifier, scaler, label_encoder,
            primary_device, seq_len=30, window_stride=15, smooth_window=15, min_segment_frames=30
        )
    logger.info(f"Found {len(segments)} segments")

    # Обработка сегментов
    all_errors = np.full((len(frames), 11), np.nan)
    all_thresholds = np.full((len(frames), 11), np.nan)
    segment_info = []

    for seg in segments:
        start_frame, end_frame, ex_name, conf = seg
        seg_indices = [i for i in range(start_frame, end_frame+1) if i in valid_indices]
        if not seg_indices:
            continue
        seg_model = ensemble_models.get(ex_name)
        score_10 = None
        score_5 = None
        thresholds = None
        errors = None

        if seg_model is not None:
            angles_seg = np.stack([all_angles[i] for i in seg_indices])
            angles_norm = (angles_seg / np.pi) * 2.0 - 1.0
            dev = model_to_device.get(ex_name, primary_device)

            # Пороги
            thresholds = get_thresholds(ex_name)
            errors = compute_joint_errors(seg_model, angles_norm, dev, seq_len=30)
            score_10 = compute_exercise_score(errors, thresholds)
            score_5 = convert_score_to_5star(score_10)

            for i, idx in enumerate(seg_indices):
                all_errors[idx] = errors[i]
                all_thresholds[idx] = thresholds

        segment_info.append({
            'start_frame': start_frame, 'end_frame': end_frame,
            'exercise': ex_name, 'confidence': float(conf),
            'score_10': score_10, 'score_5': score_5
        })

    # Генерация аннотированных кадров
    annotated_frames = []
    for i, frame in enumerate(frames):
        errs = all_errors[i]
        kpts = all_keypoints[i]
        thresh = all_thresholds[i] if not np.isnan(all_thresholds[i]).all() else None

        # Определяем текущий сегмент
        current_ex = "none"
        model_avail = False
        for seg in segment_info:
            if seg['start_frame'] <= i <= seg['end_frame']:
                current_ex = seg['exercise']
                model_avail = seg['score_10'] is not None
                break

        if np.isnan(errs).all():
            errs = np.zeros(11)
            model_avail = False

        annotated = draw_skeleton_with_errors(frame.copy(), kpts, errs, thresh, model_avail)
        # Добавляем текст на кадр
        cv2.putText(annotated, f"Ex: {current_ex}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if model_avail:
            seg = next(s for s in segment_info if s['start_frame'] <= i <= s['end_frame'])
            if seg['score_10'] is not None:
                cv2.putText(annotated, f"Score: {seg['score_10']:.1f}/10 ({seg['score_5']}★)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        annotated_frames.append(annotated)

    overall_score = np.mean([s['score_10'] for s in segment_info if s['score_10'] is not None]) if segment_info else None
    return frames, annotated_frames, segment_info, overall_score

@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = os.path.join(os.path.dirname(__file__), "web_frontend_upload.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend not found</h1>"

@app.get("/exercises")
async def list_exercises():
    return {"exercises": [{"name": name, "display_name": name.replace("_", " ").title()} for name in ensemble_models.keys()]}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...), exercise: Optional[str] = Form(None)):
    """
    Принимает видеофайл, сохраняет во временную папку, обрабатывает и возвращает:
    - video_url: ссылка на аннотированное видео
    - segments: список сегментов с оценками
    - overall_score: средняя оценка
    """
    try:
        # Сохраняем загруженный файл
        ext = os.path.splitext(file.filename)[1]
        temp_id = str(uuid.uuid4())
        input_path = f"static/temp_{temp_id}{ext}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing video: {input_path}")

        # Основная обработка (синхронная, но можно в отдельном потоке)
        frames, annotated_frames, segments, overall_score = process_uploaded_video(input_path, exercise)

        # Сохраняем аннотированное видео
        output_path = f"static/results/{temp_id}_annotated.mp4"
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
        for f in annotated_frames:
            out.write(f)
        out.release()

        # Удаляем временный файл
        os.unlink(input_path)

        return JSONResponse({
            "video_url": f"/{output_path}",
            "segments": segments,
            "overall_score": overall_score,
            "num_frames": len(frames)
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)