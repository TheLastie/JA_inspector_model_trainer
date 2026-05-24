#!/usr/bin/env python3
"""
Инференс с использованием ансамбля специализированных автоэнкодеров.
Автоматически находит все доступные модели и загружает их.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import cv2

from model_zoo import RecurrentAutoencoder, TransformerAutoencoder, LSTMVAE
from Data_maker import SplittedData
from experiment_manager import manager

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Video processing will be disabled.")


def load_encoder(model_path: str, model_type: str, params: dict, device: torch.device):
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if model_type == 'rae':
        encoder = RecurrentAutoencoder(
            input_dim=params.get('input_dim', 11),
            hidden_dim=params.get('hidden_dim', 128),
            latent_dim=params.get('latent_dim', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            autoregressive=params.get('autoregressive', True)
        )
    elif model_type == 'vae':
        encoder = LSTMVAE(
            input_dim=params.get('input_dim', 11),
            hidden_dim=params.get('hidden_dim', 128),
            latent_dim=params.get('latent_dim', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2)
        )
    elif model_type == 'transformer':
        encoder = TransformerAutoencoder(
            input_dim=params.get('input_dim', 11),
            d_model=params.get('d_model', 128),
            nhead=params.get('nhead', 4),
            num_layers=params.get('num_layers', 3),
            latent_dim=params.get('latent_dim', 64),
            seq_len=params.get('seq_len', 30),
            dropout=params.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    return encoder


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def process_video_to_angles(video_path: str, device: torch.device,
                            yolo_model_path: str = 'yolo26x-pose.pt') -> List[Optional[np.ndarray]]:
    if not YOLO_AVAILABLE:
        raise ImportError("Install ultralytics to process videos")
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


def classify_ensemble(angles_sequence: np.ndarray, ensemble_models: Dict[str, nn.Module],
                      classifier: MLPClassifier, scaler, label_encoder,
                      device: torch.device, seq_len: int = 30, stride: int = 15,
                      verbose: bool = False) -> Tuple[str, float]:
    angles_norm = (angles_sequence / np.pi) * 2.0 - 1.0
    if len(angles_norm) < seq_len:
        pad = seq_len - len(angles_norm)
        angles_norm = np.pad(angles_norm, ((0, pad), (0, 0)), mode='constant')
        windows = [angles_norm]
    else:
        windows = [angles_norm[i:i+seq_len] for i in range(0, len(angles_norm) - seq_len + 1, stride)]
        if not windows:
            windows = [angles_norm[:seq_len]]

    all_errors = []
    with torch.no_grad():
        for window in windows:
            inp = torch.FloatTensor(window).unsqueeze(0).to(device)
            window_errors = []
            for name, model in ensemble_models.items():
                if isinstance(model, LSTMVAE):
                    recon, _, _ = model(inp)
                else:
                    recon, _ = model(inp)
                mse = torch.mean((recon - inp) ** 2).item()
                window_errors.append(mse)
            all_errors.append(window_errors)

    mean_errors = np.mean(all_errors, axis=0)
    errors_norm = scaler.transform(mean_errors.reshape(1, -1))
    inp_errors = torch.FloatTensor(errors_norm).to(device)

    classifier.eval()
    with torch.no_grad():
        logits = classifier(inp_errors)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    pred_class = label_encoder.inverse_transform([pred_idx])[0]

    if verbose:
        print("Reconstruction errors:")
        for name, err in zip(ensemble_models.keys(), mean_errors):
            print(f"  {name}: {err:.6f}")
        print("Class probabilities:")
        for cls, prob in zip(label_encoder.classes_, probs):
            print(f"  {cls}: {prob:.4f}")
        print(f"Predicted: {pred_class} (confidence: {confidence:.4f})")

    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser(description='Ensemble inference with specialized autoencoders.')
    parser.add_argument('--input', required=True, help='Path to video file')
    parser.add_argument('--ensemble_path', default='models/ensemble/ensemble_classifier.pth',
                        help='Path to saved ensemble classifier')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--output_dir', default='inference_results')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--yolo_model', default='yolo26x-pose.pt')
    parser.add_argument('--classify_only', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--confidence_threshold', type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Загружаем ансамбль и классификатор
    checkpoint = torch.load(args.ensemble_path, map_location=device, weights_only=False)
    model_names = checkpoint['model_names']
    model_metadata = checkpoint['model_metadata']
    scaler = checkpoint['scaler']
    label_encoder = checkpoint['label_encoder']

    ensemble_models = {}
    for name in model_names:
        meta = model_metadata[name]
        path = os.path.join('models', manager.get_experiment_dir(meta['id']), 'best_model.pth')
        print(f"Loading {name} model from {path}")
        encoder = load_encoder(path, meta['model_type'], meta['params'], device)
        ensemble_models[name] = encoder

    num_classes = len(label_encoder.classes_)
    classifier = MLPClassifier(len(ensemble_models), num_classes)
    classifier.load_state_dict(checkpoint['classifier_state'])
    classifier.to(device)
    classifier.eval()

    # 2. Извлекаем углы из видео
    print("Extracting angles from video...")
    all_angles = process_video_to_angles(args.input, device, args.yolo_model)
    valid_angles = [a for a in all_angles if a is not None]
    if len(valid_angles) == 0:
        print("No valid frames with pose detected.")
        return
    angles_array = np.stack(valid_angles)

    # 3. Классификация
    ex_name, conf = classify_ensemble(angles_array, ensemble_models, classifier, scaler, label_encoder,
                                      device, args.seq_len, verbose=args.verbose)
    print(f"Predicted exercise: {ex_name} (confidence: {conf:.4f})")
    if conf < args.confidence_threshold:
        print(f"Warning: confidence below threshold ({args.confidence_threshold:.2f})")

    if args.classify_only:
        return

    # 4. Анализ ошибок с помощью выбранной модели
    if ex_name not in ensemble_models:
        print(f"No specialized model for {ex_name}, cannot perform anomaly detection.")
        return
    model = ensemble_models[ex_name]
    print(f"Anomaly detection for {ex_name} not yet implemented in this script.")


if __name__ == '__main__':
    main()