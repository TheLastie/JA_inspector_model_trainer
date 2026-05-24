#!/usr/bin/env python3
"""
joint_analysis.py – Покадровый анализ ошибок восстановления по каждому суставу.
Позволяет локализовать, какой именно сустав выполнил движение неправильно.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import cv2

from model_zoo import RecurrentAutoencoder, TransformerAutoencoder, LSTMVAE
from Data_maker import SplittedData
from experiment_manager import manager
from data_pipeline import load_all_frames

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Video processing will be disabled.")

JOINT_NAMES = [
    'Neck', 'R_Shoulder', 'R_Elbow', 'L_Shoulder', 'L_Elbow',
    'R_Hip', 'R_Knee', 'L_Hip', 'L_Knee', 'R_Ankle', 'L_Ankle'
]


def find_best_model_for_exercise(exercise: str, models_base_dir: str = 'models'):
    experiments = manager.list_experiments(status='completed')
    candidates = []
    for exp in experiments:
        if exp['exercise'] != exercise:
            continue
        if 'best_test_loss' not in exp.get('result', {}):
            continue
        candidates.append(exp)
    if not candidates:
        return None, None
    best_exp = min(candidates, key=lambda e: e['result']['best_test_loss'])
    exp_dir = manager.get_experiment_dir(best_exp['id'])
    model_path = os.path.join(models_base_dir, exp_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        return None, None
    return model_path, best_exp


def load_model(model_path: str, model_type: str, device: torch.device, metadata: dict) -> nn.Module:
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
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


def compute_joint_errors(model, angles_norm: np.ndarray, device: torch.device, seq_len: int = 30):
    num_frames = len(angles_norm)
    if num_frames < seq_len:
        pad = seq_len - num_frames
        angles_padded = np.pad(angles_norm, ((0, pad), (0, 0)), mode='constant')
        windows = [angles_padded]
    else:
        windows = [angles_norm[i:i+seq_len] for i in range(0, num_frames - seq_len + 1)]

    errors_per_frame = np.zeros((num_frames, 11))
    counts_per_frame = np.zeros(num_frames)

    with torch.no_grad():
        for idx, window in enumerate(windows):
            inp = torch.FloatTensor(window).unsqueeze(0).to(device)
            if isinstance(model, LSTMVAE):
                recon, _, _ = model(inp)
            else:
                recon, _ = model(inp)
            diff = torch.abs(recon - inp).cpu().numpy()[0]
            start = idx
            for t in range(len(diff)):
                frame = start + t
                if frame < num_frames:
                    errors_per_frame[frame] += diff[t]
                    counts_per_frame[frame] += 1

    errors_per_frame = errors_per_frame / np.maximum(counts_per_frame[:, np.newaxis], 1)
    return errors_per_frame


def compute_thresholds_from_normal(normal_json_path: str, model, device, seq_len: int = 30,
                                   percentile: float = 95.0) -> np.ndarray:
    angles, _ = load_all_frames(normal_json_path)
    if len(angles) == 0:
        raise ValueError("No data in normal JSON")
    angles_norm = (angles / np.pi) * 2.0 - 1.0
    errors = compute_joint_errors(model, angles_norm, device, seq_len)
    thresholds = np.percentile(errors, percentile, axis=0)
    return thresholds


def process_video(video_path: str, yolo_model_path: str, device: torch.device) -> List[Optional[np.ndarray]]:
    if not YOLO_AVAILABLE:
        raise ImportError("Install ultralytics")
    yolo = YOLO(yolo_model_path).to(device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = 600
    new_height = int(height / width * new_width)

    all_angles = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_width, new_height))
        results = yolo(resized)[0]

        angles = None
        if results.keypoints is not None and len(results.keypoints) > 0:
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                main_idx = np.argmax(areas)
            else:
                main_idx = 0

            xy = results.keypoints.xy.cpu().numpy()
            conf = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None

            if len(xy) > main_idx:
                points = {}
                for j in range(1, 14):
                    x, y = xy[main_idx, j]
                    c = conf[main_idx, j] if conf is not None else 1.0
                    points[j] = [(x, y), c]
                try:
                    splitted = SplittedData(points)
                    angles = np.array(splitted.angles, dtype=np.float32)
                except:
                    angles = None
        all_angles.append(angles)
    cap.release()
    return all_angles


def plot_joint_heatmap(errors: np.ndarray, thresholds: Optional[np.ndarray] = None,
                       output_path: str = 'joint_heatmap.png'):
    plt.figure(figsize=(14, 6))
    sns.heatmap(errors.T, cmap='hot', cbar_kws={'label': 'Reconstruction Error'})
    plt.xlabel('Frame')
    plt.ylabel('Joint')
    plt.yticks(ticks=np.arange(len(JOINT_NAMES)) + 0.5, labels=JOINT_NAMES, rotation=0)
    plt.title('Joint-wise Reconstruction Error')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_path}")


def plot_error_bars(mean_errors: np.ndarray, thresholds: Optional[np.ndarray] = None,
                    output_path: str = 'joint_errors_bar.png'):
    plt.figure(figsize=(12, 5))
    x = np.arange(len(JOINT_NAMES))
    plt.bar(x, mean_errors, color='steelblue', alpha=0.7, label='Mean Error')
    if thresholds is not None:
        plt.plot(x, thresholds, 'r--', label='Threshold', linewidth=2)
    plt.xticks(x, JOINT_NAMES, rotation=45, ha='right')
    plt.ylabel('Mean Absolute Error')
    plt.title('Average Joint Reconstruction Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Bar plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Joint-wise anomaly analysis.')
    parser.add_argument('--model_path', help='Path to autoencoder .pth')
    parser.add_argument('--exercise', help='Exercise name (auto-select best model)')
    parser.add_argument('--model_type', choices=['rae', 'transformer', 'vae'],
                        help='Required if --model_path is used')
    parser.add_argument('--input', required=True, help='Video file or JSON')
    parser.add_argument('--input_type', choices=['video', 'json'], default='video')
    parser.add_argument('--normal_json', help='JSON with "normal" examples to compute thresholds. '
                                              'If not provided, uses the original training JSON for the exercise.')
    parser.add_argument('--no_thresholds', action='store_true',
                        help='Do not compute or use thresholds (just show mean errors)')
    parser.add_argument('--threshold_percentile', type=float, default=95.0)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--output_dir', default='joint_analysis')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--yolo_model', default='yolo26x-pose.pt')
    parser.add_argument('--save_errors', action='store_true', help='Save errors array to .npy')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Определение модели
    if args.model_path:
        if not args.model_type:
            parser.error("--model_type required with --model_path")
        model_path = args.model_path
        model_type = args.model_type
        metadata = {}
        exercise_name = None
    else:
        if not args.exercise:
            parser.error("Either --model_path or --exercise required")
        exercise_name = args.exercise
        model_path, metadata = find_best_model_for_exercise(exercise_name)
        if model_path is None:
            raise RuntimeError(f"No model found for exercise '{exercise_name}'")
        model_type = metadata['model_type']
        print(f"Found best model for '{exercise_name}': {model_type} (loss={metadata['result']['best_test_loss']:.6f})")

    model = load_model(model_path, model_type, device, metadata)

    # Определение источника для порогов
    thresholds = None
    if not args.no_thresholds:
        if args.normal_json:
            normal_source = args.normal_json
        elif exercise_name:
            # Ищем JSON с именем упражнения в datasets/
            candidate = os.path.join('datasets', f'{exercise_name}.json')
            if os.path.exists(candidate):
                normal_source = candidate
                print(f"Using training data {candidate} to compute thresholds.")
            else:
                print(f"Warning: No normal_json provided and {candidate} not found. Thresholds disabled.")
                normal_source = None
        else:
            print("Warning: Cannot determine normal JSON without exercise name. Thresholds disabled.")
            normal_source = None

        if normal_source:
            print(f"Computing thresholds from {normal_source}...")
            thresholds = compute_thresholds_from_normal(normal_source, model, device,
                                                        args.seq_len, args.threshold_percentile)
            print("Thresholds per joint (normalized scale):")
            for i, name in enumerate(JOINT_NAMES):
                print(f"  {name}: {thresholds[i]:.6f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Загрузка входных данных
    if args.input_type == 'json':
        angles, _ = load_all_frames(args.input)
        if len(angles) == 0:
            raise ValueError("No data in JSON")
        angles_norm = (angles / np.pi) * 2.0 - 1.0
    else:
        all_angles = process_video(args.input, args.yolo_model, device)
        valid_angles = [a for a in all_angles if a is not None]
        if not valid_angles:
            print("No valid frames with pose detected.")
            return
        angles_norm = np.stack(valid_angles)
        angles_norm = (angles_norm / np.pi) * 2.0 - 1.0

    print(f"Processing {len(angles_norm)} frames...")
    errors = compute_joint_errors(model, angles_norm, device, args.seq_len)

    if args.save_errors:
        np.save(os.path.join(args.output_dir, 'joint_errors.npy'), errors)
        print(f"Errors saved to {args.output_dir}/joint_errors.npy")

    mean_errors = np.mean(errors, axis=0)

    # Анализ
    if thresholds is not None:
        anomaly_mask = errors > thresholds
        anomalous_frames, anomalous_joints = np.where(anomaly_mask)
        print(f"Found {len(anomalous_frames)} anomalous joint-frame pairs.")
        joint_anomaly_counts = np.sum(anomaly_mask, axis=0)
        print("Anomalous frames per joint:")
        for i, name in enumerate(JOINT_NAMES):
            print(f"  {name}: {joint_anomaly_counts[i]}")
    else:
        print("No thresholds used. Mean errors per joint:")
        for i, name in enumerate(JOINT_NAMES):
            print(f"  {name}: {mean_errors[i]:.6f}")
        joint_anomaly_counts = None

    # Графики
    plot_joint_heatmap(errors, thresholds, os.path.join(args.output_dir, 'joint_heatmap.png'))
    plot_error_bars(mean_errors, thresholds, os.path.join(args.output_dir, 'joint_errors_bar.png'))

    # Сохранение результатов
    results = {
        'thresholds': thresholds.tolist() if thresholds is not None else None,
        'mean_errors': mean_errors.tolist(),
        'joint_names': JOINT_NAMES,
        'anomaly_counts': joint_anomaly_counts.tolist() if joint_anomaly_counts is not None else None,
    }
    with open(os.path.join(args.output_dir, 'analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("Analysis complete.")


if __name__ == '__main__':
    main()