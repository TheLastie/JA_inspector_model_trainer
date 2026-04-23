#!/usr/bin/env python3
"""
joint_analysis_video.py – Ускоренный анализ видео с сегментацией, уверенностью и 5-звёздной оценкой.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
import joblib
import xgboost as xgb
from tqdm import tqdm

from model_zoo import RecurrentAutoencoder, TransformerAutoencoder, LSTMVAE
from Data_maker import SplittedData
from experiment_manager import manager
from data_pipeline import load_all_frames

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: ultralytics not installed.")
    sys.exit(1)

JOINT_NAMES = [
    'Neck', 'R_Shoulder', 'R_Elbow', 'L_Shoulder', 'L_Elbow',
    'R_Hip', 'R_Knee', 'L_Hip', 'L_Knee', 'R_Ankle', 'L_Ankle'
]

SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

ANGLE_TO_EDGES = {
    0: [(5,6), (11,12)], 1: [(6,8), (8,10)], 2: [(8,10)],
    3: [(5,7), (7,9)], 4: [(7,9)], 5: [(12,14), (14,16)],
    6: [(14,16)], 7: [(11,13), (13,15)], 8: [(13,15)],
    9: [(14,16)], 10: [(13,15)]
}

ANGLE_TO_KEYPOINTS = {
    0: [5,6,11,12], 1: [6,8,10], 2: [8,10], 3: [5,7,9], 4: [7,9],
    5: [12,14,16], 6: [14,16], 7: [11,13,15], 8: [13,15], 9: [14,16], 10: [13,15]
}

# ----- Загрузка моделей автоэнкодеров -----
def load_model(model_path: str, model_type: str, device: torch.device, metadata: dict) -> nn.Module:
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    params = metadata.get('params', {})

    if model_type == 'rae':
        model = RecurrentAutoencoder(
            input_dim=params.get('input_dim', 11),
            hidden_dim=params.get('hidden_dim', 128),
            latent_dim=params.get('latent_dim', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            autoregressive=params.get('autoregressive', True)
        )
    elif model_type == 'vae':
        model = LSTMVAE(
            input_dim=params.get('input_dim', 11),
            hidden_dim=params.get('hidden_dim', 128),
            latent_dim=params.get('latent_dim', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2)
        )
    elif model_type == 'transformer':
        model = TransformerAutoencoder(
            input_dim=params.get('input_dim', 11),
            d_model=params.get('d_model', 128),
            nhead=params.get('nhead', 4),
            num_layers=params.get('num_layers', 3),
            latent_dim=params.get('latent_dim', 64),
            seq_len=params.get('seq_len', 30),
            dropout=params.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ----- Загрузка ансамбля -----
class XGBWrapper:
    def __init__(self, bst, scaler, label_encoder, model_names):
        self.bst = bst
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.model_names = model_names

    def predict_proba(self, features):
        dmatrix = xgb.DMatrix(features)
        probs = self.bst.predict(dmatrix, output_margin=False)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        return probs


def load_ensemble(ensemble_dir: str, device: torch.device, quiet: bool = False):
    xgb_path = os.path.join(ensemble_dir, 'ensemble_classifier_xgb.json')
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")

    bst = xgb.Booster()
    bst.load_model(xgb_path)
    scaler = joblib.load(os.path.join(ensemble_dir, 'scaler.joblib'))
    label_encoder = joblib.load(os.path.join(ensemble_dir, 'label_encoder.joblib'))
    with open(os.path.join(ensemble_dir, 'model_metadata.json'), 'r') as f:
        meta = json.load(f)
    model_names = meta['model_names']
    models_meta = meta['model_metadata']

    ensemble_models = {}
    for name in tqdm(model_names, desc="Loading models", disable=quiet):
        meta = models_meta[name]
        exp_dir = manager.get_experiment_dir(meta['id'])
        model_path = os.path.join('models', exp_dir, 'best_model.pth')
        encoder = load_model(model_path, meta['model_type'], device, meta)
        ensemble_models[name] = encoder

    classifier = XGBWrapper(bst, scaler, label_encoder, model_names)
    return ensemble_models, classifier, scaler, label_encoder


# ----- Быстрое вычисление признаков для окна -----
def compute_window_features(window: np.ndarray, ensemble_models: Dict[str, nn.Module], device: torch.device):
    inp = torch.FloatTensor(window).unsqueeze(0).to(device)
    feats = []
    model_errors = {}
    with torch.no_grad():
        for name, model in ensemble_models.items():
            if isinstance(model, LSTMVAE):
                recon, _, _ = model(inp)
            else:
                recon, _ = model(inp)
            diff = recon - inp
            mse = torch.mean(diff ** 2).item()
            mae = torch.mean(torch.abs(diff)).item()
            max_err = torch.max(torch.abs(diff)).item()
            var_mse = torch.var(torch.mean(diff ** 2, dim=2)).item()
            feats.extend([mse, mae, max_err, var_mse])
            model_errors[name] = mse
    num_models = len(ensemble_models)
    for i in range(num_models):
        for j in range(i+1, num_models):
            ratio = feats[i*4] / (feats[j*4] + 1e-8)
            feats.append(ratio)
    return np.array(feats), model_errors


# ----- Классификация окна -----
def classify_ensemble_window(angles_norm: np.ndarray, ensemble_models: Dict[str, nn.Module],
                             classifier: XGBWrapper, scaler, label_encoder,
                             device: torch.device, seq_len: int = 30):
    if len(angles_norm) < seq_len:
        pad = seq_len - len(angles_norm)
        angles_norm = np.pad(angles_norm, ((0, pad), (0, 0)), mode='constant')
        window = angles_norm
    else:
        start = (len(angles_norm) - seq_len) // 2
        window = angles_norm[start:start+seq_len]

    feats, model_errors = compute_window_features(window, ensemble_models, device)
    feats_norm = scaler.transform(feats.reshape(1, -1))
    probs = classifier.predict_proba(feats_norm)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    pred_class = label_encoder.inverse_transform([pred_idx])[0]
    return pred_class, confidence, probs, model_errors


# ----- Сегментация видео с уверенностью -----
def segment_video_simple(angles_list: List[Optional[np.ndarray]], valid_indices: List[int],
                         ensemble_models, classifier, scaler, label_encoder,
                         device: torch.device, seq_len: int = 30, window_stride: int = 15,
                         smooth_window: int = 15, min_segment_frames: int = 30):
    if not valid_indices:
        return []

    angles_array = np.stack([angles_list[i] for i in valid_indices])
    angles_norm = (angles_array / np.pi) * 2.0 - 1.0

    window_results = []
    for start in range(0, len(angles_norm) - seq_len + 1, window_stride):
        window = angles_norm[start:start+seq_len]
        pred_class, conf, probs, _ = classify_ensemble_window(
            window, ensemble_models, classifier, scaler, label_encoder, device, seq_len
        )
        window_results.append((start, start+seq_len, pred_class, conf))

    if not window_results:
        pred_class, conf, probs, _ = classify_ensemble_window(
            angles_norm, ensemble_models, classifier, scaler, label_encoder, device, seq_len
        )
        return [(0, len(valid_indices)-1, pred_class, conf)]

    frame_classes = np.full(len(angles_norm), 'unknown', dtype=object)
    frame_conf = np.zeros(len(angles_norm))
    for start, end, cls, conf in window_results:
        frame_classes[start:end] = cls
        frame_conf[start:end] = np.maximum(frame_conf[start:end], conf)

    # Сглаживание
    smoothed = np.copy(frame_classes)
    half = smooth_window // 2
    for i in range(len(frame_classes)):
        left = max(0, i - half)
        right = min(len(frame_classes), i + half + 1)
        window = frame_classes[left:right]
        if len(window) > 0:
            counter = Counter(window)
            smoothed[i] = counter.most_common(1)[0][0]

    # Формирование сегментов
    segments = []
    current_cls = smoothed[0]
    start = 0
    for i in range(1, len(smoothed)):
        if smoothed[i] != current_cls:
            seg_conf = np.mean(frame_conf[start:i])
            segments.append((start, i-1, current_cls, seg_conf))
            current_cls = smoothed[i]
            start = i
    seg_conf = np.mean(frame_conf[start:len(smoothed)])
    segments.append((start, len(smoothed)-1, current_cls, seg_conf))

    # Удаление коротких сегментов
    merged = []
    i = 0
    while i < len(segments):
        s, e, cls, conf = segments[i]
        if e - s + 1 >= min_segment_frames:
            merged.append((s, e, cls, conf))
            i += 1
        else:
            if i > 0:
                prev_s, prev_e, prev_cls, prev_conf = merged[-1]
                merged[-1] = (prev_s, e, prev_cls, (prev_conf + conf) / 2)
            elif i + 1 < len(segments):
                next_s, next_e, next_cls, next_conf = segments[i+1]
                segments[i+1] = (s, next_e, next_cls, (conf + next_conf) / 2)
            i += 1

    final_segments = []
    for s_rel, e_rel, cls, conf in merged:
        final_segments.append((valid_indices[s_rel], valid_indices[e_rel], cls, conf))
    return final_segments


# ----- Вычисление ошибок по суставам -----
def compute_joint_errors(model, angles_norm: np.ndarray, device: torch.device, seq_len: int = 30):
    num_frames = len(angles_norm)
    stride = max(1, seq_len // 2)
    if num_frames < seq_len:
        pad = seq_len - num_frames
        angles_padded = np.pad(angles_norm, ((0, pad), (0, 0)), mode='constant')
        windows = [angles_padded]
    else:
        windows = [angles_norm[i:i+seq_len] for i in range(0, num_frames - seq_len + 1, stride)]

    errors_per_frame = np.zeros((num_frames, 11))
    counts = np.zeros(num_frames)

    with torch.no_grad():
        for idx, window in enumerate(windows):
            inp = torch.FloatTensor(window).unsqueeze(0).to(device)
            if isinstance(model, LSTMVAE):
                recon, _, _ = model(inp)
            else:
                recon, _ = model(inp)
            diff = torch.abs(recon - inp).cpu().numpy()[0]
            start = idx * stride
            for t in range(len(diff)):
                frame = start + t
                if frame < num_frames:
                    errors_per_frame[frame] += diff[t]
                    counts[frame] += 1

    errors_per_frame /= np.maximum(counts[:, np.newaxis], 1)
    return errors_per_frame


def compute_exercise_score(errors: np.ndarray, thresholds: np.ndarray) -> float:
    if errors.size == 0 or np.isnan(errors).all():
        return 0.0
    exceed_ratio = np.mean(errors > thresholds, axis=0)
    exceed_magnitude = np.maximum(errors - thresholds, 0).mean(axis=0) / (thresholds + 1e-8)
    penalties = exceed_ratio + 0.5 * exceed_magnitude
    avg_penalty = np.mean(penalties)
    score = 10.0 * max(0.0, 1.0 - avg_penalty)
    return round(score, 1)


def convert_score_to_5star(score_10: float) -> int:
    """Преобразует 10-балльную оценку 8..10 в 1..5 звёзд."""
    if score_10 < 8.0:
        return 1
    elif score_10 < 8.5:
        return 2
    elif score_10 < 9.0:
        return 3
    elif score_10 < 9.5:
        return 4
    else:
        return 5


def find_reference_json(exercise: str, data_dir: str = 'datasets') -> Optional[str]:
    json_path = os.path.join(data_dir, f"{exercise}.json")
    if os.path.exists(json_path):
        return json_path
    alt_name = exercise.replace('-', '_')
    json_path = os.path.join(data_dir, f"{alt_name}.json")
    if os.path.exists(json_path):
        return json_path
    return None


# ----- Обработка видео YOLO -----
def process_video_with_yolo(video_path: str, yolo_model_path: str, device: torch.device, quiet: bool = False):
    import logging
    if quiet:
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
    yolo = YOLO(yolo_model_path).to(device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = 600
    new_height = int(height / width * new_width)

    frames = []
    all_angles = []
    all_keypoints = []
    all_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_width, new_height))
        results = yolo(resized, verbose=False)[0]

        angles = None
        keypoints = None
        boxes = None

        if results.keypoints is not None and len(results.keypoints) > 0:
            if results.boxes is not None and len(results.boxes) > 0:
                boxes_arr = results.boxes.xyxy.cpu().numpy()
                areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
                main_idx = np.argmax(areas)
                boxes = boxes_arr[main_idx]
            else:
                main_idx = 0

            xy = results.keypoints.xy.cpu().numpy()
            conf = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None

            if len(xy) > main_idx:
                keypoints = xy[main_idx]
                points = {}
                for j in range(1, 14):
                    x, y = keypoints[j]
                    c = conf[main_idx, j] if conf is not None else 1.0
                    points[j] = [(x, y), c]
                try:
                    splitted = SplittedData(points)
                    angles = np.array(splitted.angles, dtype=np.float32)
                except:
                    angles = None

        frames.append(resized)
        all_angles.append(angles)
        all_keypoints.append(keypoints)
        all_boxes.append(boxes)

    cap.release()
    return frames, all_angles, all_keypoints, all_boxes


def draw_skeleton_with_errors(frame, keypoints, joint_errors, thresholds, model_available=True):
    if keypoints is None:
        return frame

    if not model_available:
        for edge in SKELETON_EDGES:
            pt1 = tuple(keypoints[edge[0]].astype(int))
            pt2 = tuple(keypoints[edge[1]].astype(int))
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame, pt1, pt2, (128, 128, 128), 2)
        for kpt in keypoints:
            pt = tuple(kpt.astype(int))
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(frame, pt, 3, (128, 128, 128), -1)
        return frame

    angle_colors = []
    for i, err in enumerate(joint_errors):
        if thresholds is not None and err > thresholds[i]:
            angle_colors.append((0, 0, 255))
        elif thresholds is not None and err > 0.7 * thresholds[i]:
            angle_colors.append((0, 255, 255))
        else:
            angle_colors.append((0, 255, 0))

    edge_colors = {}
    for angle_idx, edges in ANGLE_TO_EDGES.items():
        color = angle_colors[angle_idx]
        for edge in edges:
            edge_colors[edge] = color

    for edge in SKELETON_EDGES:
        color = edge_colors.get(edge, (0, 255, 0))
        pt1 = tuple(keypoints[edge[0]].astype(int))
        pt2 = tuple(keypoints[edge[1]].astype(int))
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(frame, pt1, pt2, color, 2)

    for angle_idx, kpt_indices in ANGLE_TO_KEYPOINTS.items():
        color = angle_colors[angle_idx]
        for idx in kpt_indices:
            if idx < len(keypoints):
                pt = tuple(keypoints[idx].astype(int))
                if pt[0] > 0 and pt[1] > 0:
                    cv2.circle(frame, pt, 4, color, -1)

    return frame


# ----- MAIN -----
def main():
    parser = argparse.ArgumentParser(description='Joint-wise video analysis with confidence and 5-star rating.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--ensemble_dir', default='models/ensemble')
    parser.add_argument('--single_exercise', help='Manual exercise name (skip classification)')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--output_dir', default='joint_analysis_output')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ID to use (e.g., 0)')
    parser.add_argument('--yolo_model', default='yolo26x-pose.pt')
    parser.add_argument('--threshold_percentile', type=float, default=95.0)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    if not args.quiet:
        print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Обработка видео YOLO
    if not args.quiet:
        print("Processing video with YOLO...")
    frames, all_angles, all_keypoints, all_boxes = process_video_with_yolo(
        args.input, args.yolo_model, device, args.quiet
    )

    valid_indices = [i for i, a in enumerate(all_angles) if a is not None]
    if not valid_indices:
        print("No valid frames with pose detected.")
        return

    # 2. Определение сегментов
    if args.single_exercise:
        model_path, meta = find_best_model_for_exercise(args.single_exercise)
        if model_path is None:
            raise RuntimeError(f"No model for {args.single_exercise}")
        model = load_model(model_path, meta['model_type'], device, meta)
        segments = [(valid_indices[0], valid_indices[-1], args.single_exercise, 1.0)]
        ensemble_models = {args.single_exercise: model}
    else:
        ensemble_models, classifier, scaler, label_encoder = load_ensemble(args.ensemble_dir, device, args.quiet)
        if not args.quiet:
            print(f"Loaded {len(ensemble_models)} models. Segmenting video...")
        segments = segment_video_simple(
            all_angles, valid_indices, ensemble_models, classifier, scaler, label_encoder,
            device, args.seq_len
        )
        print(f"Found {len(segments)} segments:")
        for seg in segments:
            print(f"  Frames {seg[0]}-{seg[1]}: {seg[2]} (conf={seg[3]:.2f})")

    # 3. Последовательная обработка сегментов
    all_errors = np.full((len(frames), 11), np.nan)
    all_thresholds = np.full((len(frames), 11), np.nan)
    segment_info = []

    for seg in tqdm(segments, desc="Processing segments", disable=args.quiet):
        start_frame, end_frame, ex_name, conf = seg
        seg_indices = [i for i in range(start_frame, end_frame+1) if i in valid_indices]
        if not seg_indices:
            continue

        seg_model = ensemble_models.get(ex_name)
        score = None
        score_5 = None
        thresholds = None
        errors = None

        if seg_model is not None:
            angles_seg = np.stack([all_angles[i] for i in seg_indices])
            angles_norm = (angles_seg / np.pi) * 2.0 - 1.0

            ref_json = find_reference_json(ex_name)
            if ref_json and os.path.exists(ref_json):
                ref_angles, _ = load_all_frames(ref_json)
                if len(ref_angles) > 0:
                    ref_norm = (ref_angles / np.pi) * 2.0 - 1.0
                    ref_errors = compute_joint_errors(seg_model, ref_norm, device, args.seq_len)
                    thresholds = np.percentile(ref_errors, args.threshold_percentile, axis=0)
            if thresholds is None:
                thresholds = np.percentile(compute_joint_errors(seg_model, angles_norm, device, args.seq_len),
                                           args.threshold_percentile, axis=0)
            errors = compute_joint_errors(seg_model, angles_norm, device, args.seq_len)
            score = compute_exercise_score(errors, thresholds)
            score_5 = convert_score_to_5star(score)

            for i, idx in enumerate(seg_indices):
                all_errors[idx] = errors[i]
                all_thresholds[idx] = thresholds

        segment_info.append((start_frame, end_frame, ex_name, seg_model is not None, score, score_5, conf))
        if not args.quiet:
            print(f"  Segment {start_frame}-{end_frame} ({ex_name}): Score {score}/10 ({score_5}★/5), conf={conf:.2f}")

    # 4. Сохранение видео
    if args.save_video:
        out_path = os.path.join(args.output_dir, 'annotated_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(out_path, fourcc, 30, (w, h))

        current_ex = "none"
        current_score = None
        current_stars = None
        for i, frame in enumerate(frames):
            errs = all_errors[i]
            kpts = all_keypoints[i]
            thresh = all_thresholds[i] if not np.isnan(all_thresholds[i]).all() else None

            for seg in segment_info:
                if seg[0] <= i <= seg[1]:
                    current_ex = seg[2]
                    model_avail = seg[3]
                    current_score = seg[4]
                    current_stars = seg[5]
                    break

            if np.isnan(errs).all():
                errs = np.zeros(11)
                model_avail = False

            annotated = draw_skeleton_with_errors(frame.copy(), kpts, errs, thresh, model_avail)
            cv2.putText(annotated, f"Ex: {current_ex}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if model_avail and current_score is not None:
                cv2.putText(annotated, f"Score: {current_score}/10 ({current_stars}★/5)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            out.write(annotated)
        out.release()
        print(f"Annotated video saved to {out_path}")

    np.save(os.path.join(args.output_dir, 'joint_errors.npy'), all_errors)
    np.save(os.path.join(args.output_dir, 'thresholds.npy'), all_thresholds)
    with open(os.path.join(args.output_dir, 'segments.json'), 'w') as f:
        json.dump([{'start': s[0], 'end': s[1], 'exercise': s[2], 'score_10': s[4], 'score_5': s[5], 'confidence': s[6]} for s in segment_info], f, indent=2)

    print("Analysis complete.")


if __name__ == '__main__':
    main()