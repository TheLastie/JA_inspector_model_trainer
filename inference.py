#!/usr/bin/env python3
"""
inference.py – Анализ движений с помощью лучшей обученной модели автоэнкодера.
Также включает классификатор упражнений для автоматического определения типа движения.
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

_exercise_classifier = None
_label_encoder = None

EXERCISE_CLASSES = ['squat', 'shoulder_press', 'push-up', 'pull-up', 'other']


class ExerciseClassifier(nn.Module):
    """Классификатор упражнений на основе энкодера."""
    def __init__(self, encoder, latent_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            if isinstance(self.encoder, LSTMVAE):
                mu, _ = self.encoder.encode(x)
                latent = mu
            else:
                _, latent = self.encoder(x)
        return self.classifier(latent)


def find_best_model(exercise: str, model_type: Optional[str] = None,
                    models_base_dir: str = 'models') -> Tuple[str, str, Dict[str, Any]]:
    experiments = manager.list_experiments(status='completed')
    candidates = []
    for exp in experiments:
        if exp['exercise'] != exercise:
            continue
        if model_type is not None and exp['model_type'] != model_type:
            continue
        if 'best_test_loss' not in exp.get('result', {}):
            continue
        candidates.append(exp)

    if not candidates:
        raise ValueError(f"No completed experiments found for exercise '{exercise}'"
                         + (f" with model_type '{model_type}'" if model_type else ""))

    best_exp = min(candidates, key=lambda e: e['result']['best_test_loss'])
    exp_dir = manager.get_experiment_dir(best_exp['id'])
    model_path = os.path.join(models_base_dir, exp_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return model_path, best_exp['model_type'], best_exp


def load_model(model_path: str, model_type: str, device: torch.device,
               metadata: Optional[Dict[str, Any]] = None) -> nn.Module:
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    params = metadata.get('params', {}) if metadata else {}

    if model_type in ('rae', 'vae'):
        encoder_layers = set()
        for key in state_dict.keys():
            if key.startswith('encoder_lstm.weight_ih_l'):
                encoder_layers.add(int(key.split('_l')[-1]))
        num_layers = max(len(encoder_layers), 1)

        if 'encoder_fc.weight' in state_dict:
            latent_dim = state_dict['encoder_fc.weight'].shape[0]
            hidden_dim = state_dict['encoder_fc.weight'].shape[1]
        else:
            latent_dim = params.get('latent_dim', 64)
            hidden_dim = params.get('hidden_dim', 128)

        input_dim = state_dict.get('decoder_out.weight', torch.empty(11, hidden_dim)).shape[0]
        dropout = params.get('dropout', 0.2)

        if model_type == 'rae':
            model = RecurrentAutoencoder(input_dim, hidden_dim, latent_dim, num_layers, dropout,
                                         autoregressive=params.get('autoregressive', True))
        else:
            model = LSTMVAE(input_dim, hidden_dim, latent_dim, num_layers, dropout)

    elif model_type == 'transformer':
        if 'input_proj.weight' in state_dict:
            d_model = state_dict['input_proj.weight'].shape[0]
            input_dim = state_dict['input_proj.weight'].shape[1]
        else:
            d_model = params.get('d_model', 128)
            input_dim = params.get('input_dim', 11)
        latent_dim = state_dict.get('latent_fc.weight', torch.empty(64, d_model)).shape[0]
        nhead = params.get('nhead', 4)
        num_layers = params.get('num_layers', 3)
        dropout = params.get('dropout', 0.1)
        model = TransformerAutoencoder(input_dim, d_model, nhead, num_layers, latent_dim,
                                       params.get('seq_len', 30), dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_classifier(classifier_path: str, device: torch.device):
    global _exercise_classifier, _label_encoder
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
    encoder_exp_id = checkpoint['encoder_exp_id']
    exp = manager.load_experiment(encoder_exp_id)
    if exp is None:
        raise ValueError(f"Encoder experiment {encoder_exp_id} not found")
    model_type = exp['model_type']
    params = exp['params']
    exp_dir = manager.get_experiment_dir(exp['id'])
    encoder_path = os.path.join('models', exp_dir, 'best_model.pth')
    encoder = load_model(encoder_path, model_type, device, exp)
    latent_dim = params.get('latent_dim', 64)
    num_classes = len(checkpoint['label_encoder'].classes_)
    classifier = ExerciseClassifier(encoder, latent_dim, num_classes=num_classes)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier.eval()
    _exercise_classifier = classifier
    _label_encoder = checkpoint['label_encoder']
    print(f"Loaded classifier from {classifier_path}")


def classify_exercise(angles_sequence: np.ndarray, device: torch.device,
                     seq_len: int = 30, stride: int = 15, verbose: bool = False) -> Tuple[str, float]:
    if _exercise_classifier is None:
        raise RuntimeError("Classifier not loaded.")
    angles_norm = (angles_sequence / np.pi) * 2.0 - 1.0
    if len(angles_norm) < seq_len:
        pad = seq_len - len(angles_norm)
        angles_norm = np.pad(angles_norm, ((0, pad), (0, 0)), mode='constant')
        windows = [angles_norm]
    else:
        windows = [angles_norm[i:i+seq_len] for i in range(0, len(angles_norm) - seq_len + 1, stride)]
        if not windows:
            windows = [angles_norm[:seq_len]]

    all_probs = []
    _exercise_classifier.eval()
    with torch.no_grad():
        for window in windows:
            inp = torch.FloatTensor(window).unsqueeze(0).to(device)
            logits = _exercise_classifier(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    mean_probs = np.mean(all_probs, axis=0)
    pred_idx = np.argmax(mean_probs)
    confidence = mean_probs[pred_idx]
    pred_class = _label_encoder.inverse_transform([pred_idx])[0]

    if verbose:
        print("Class probabilities:")
        for cls, prob in zip(_label_encoder.classes_, mean_probs):
            print(f"  {cls}: {prob:.4f}")
        print(f"Predicted: {pred_class} (confidence: {confidence:.4f})")

    return pred_class, confidence


def process_json_file(json_path: str, model: nn.Module, device: torch.device,
                     seq_len: int = 30, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    from data_pipeline import load_all_frames
    angles, _ = load_all_frames(json_path)
    if len(angles) == 0:
        raise ValueError("No data in JSON")

    angles_norm = (angles / np.pi) * 2.0 - 1.0 if normalize else angles.copy()
    num_frames = len(angles_norm)

    if num_frames < seq_len:
        pad = seq_len - num_frames
        angles_padded = np.pad(angles_norm, ((0, pad), (0, 0)), mode='constant')
        windows = [angles_padded]
        valid_lengths = [num_frames]
    else:
        windows = [angles_norm[i:i+seq_len] for i in range(0, num_frames - seq_len + 1)]

    errors_per_frame = np.zeros(num_frames)
    counts_per_frame = np.zeros(num_frames)

    with torch.no_grad():
        for idx, window in enumerate(windows):
            inp = torch.FloatTensor(window).unsqueeze(0).to(device)
            if isinstance(model, LSTMVAE):
                recon, _, _ = model(inp)
            else:
                recon, _ = model(inp)
            mse = torch.mean((recon - inp) ** 2, dim=2).cpu().numpy().flatten()
            start = idx
            for t in range(len(mse)):
                if start + t < num_frames:
                    errors_per_frame[start + t] += mse[t]
                    counts_per_frame[start + t] += 1

    errors_per_frame /= np.maximum(counts_per_frame, 1)
    return errors_per_frame, angles


def process_video_file(video_path: str, model: Optional[nn.Module], device: torch.device,
                       seq_len: int = 30, normalize: bool = True,
                       yolo_model_path: str = 'yolo26x-pose.pt',
                       return_angles: bool = False) -> Tuple[np.ndarray, List]:
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

    buffer = []
    frame_errors = []
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
        if model is not None and not return_angles:
            if angles is not None:
                buffer.append(angles)
                if len(buffer) >= seq_len:
                    window = np.stack(buffer[-seq_len:], axis=0)
                    if normalize:
                        window = (window / np.pi) * 2.0 - 1.0
                    inp = torch.FloatTensor(window).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if isinstance(model, LSTMVAE):
                            recon, _, _ = model(inp)
                        else:
                            recon, _ = model(inp)
                        mse = torch.mean((recon - inp) ** 2, dim=2).cpu().numpy().flatten()
                    frame_errors.append(mse[-1])
                else:
                    frame_errors.append(0.0)
            else:
                frame_errors.append(np.nan)

    cap.release()
    if model is None or return_angles:
        return np.array([]), all_angles
    return np.array(frame_errors), all_angles


def plot_errors(errors: np.ndarray, threshold: Optional[float] = None, output_path: str = 'error_plot.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(errors, 'b-', alpha=0.7, label='Reconstruction Error')
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.xlabel('Frame')
    plt.ylabel('MSE')
    plt.title('Anomaly Detection via Reconstruction Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"График сохранён в {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Inference with autoencoder and optional exercise classifier.')
    parser.add_argument('--model_path', help='Explicit path to .pth file (requires --model_type)')
    parser.add_argument('--exercise', help='Exercise name (e.g., squat) – auto-select best model')
    parser.add_argument('--model_type', choices=['rae', 'transformer', 'vae'],
                        help='Model architecture (required if --model_path is used)')
    parser.add_argument('--input', required=True, help='Path to JSON or video file')
    parser.add_argument('--input_type', choices=['json', 'video'], default='json')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--threshold_percentile', type=float, default=95.0)
    parser.add_argument('--output_dir', default='inference_results')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--yolo_model', default='yolo26x-pose.pt')
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--models_base_dir', default='models')
    parser.add_argument('--classifier_path', help='Path to trained exercise classifier .pth')
    parser.add_argument('--classify_only', action='store_true',
                        help='Only classify exercise, do not compute errors')
    parser.add_argument('--verbose', action='store_true', help='Show class probabilities during classification')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence to accept classification (if below, marks as uncertain)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.classifier_path:
        load_classifier(args.classifier_path, device)

    if args.classify_only:
        if args.classifier_path is None:
            parser.error("--classifier_path required for --classify_only")
        if args.input_type != 'video':
            raise NotImplementedError("Classification from JSON not implemented yet. Use video.")
        print("Extracting angles from video...")
        _, all_angles = process_video_file(
            args.input, None, device, args.seq_len, not args.no_normalize, args.yolo_model, return_angles=True
        )
        valid_angles = [a for a in all_angles if a is not None]
        if len(valid_angles) == 0:
            print("No valid frames with pose detected.")
            return
        angles_array = np.stack(valid_angles)
        ex_name, conf = classify_exercise(angles_array, device, args.seq_len, verbose=args.verbose)
        print(f"Predicted exercise: {ex_name} (confidence: {conf:.4f})")
        if conf < args.confidence_threshold:
            print(f"Warning: confidence below threshold ({args.confidence_threshold:.2f})")
        return

    exercise = args.exercise
    model_path = args.model_path
    model_type = args.model_type
    metadata = None

    if model_path is None and exercise is None:
        if args.classifier_path is None:
            parser.error("Either --model_path, --exercise, or --classifier_path (for auto-detection) must be provided.")
        if args.input_type != 'video':
            raise NotImplementedError("Auto-detection from JSON not supported. Please specify --exercise manually.")
        print("No exercise or model specified, using classifier to detect exercise...")
        _, all_angles = process_video_file(
            args.input, None, device, args.seq_len, not args.no_normalize, args.yolo_model, return_angles=True
        )
        valid_angles = [a for a in all_angles if a is not None]
        if len(valid_angles) == 0:
            print("Cannot classify: no pose detected.")
            return
        angles_array = np.stack(valid_angles)
        exercise, conf = classify_exercise(angles_array, device, args.seq_len, verbose=args.verbose)
        print(f"Auto-detected exercise: {exercise} (confidence: {conf:.4f})")
        if conf < args.confidence_threshold:
            print(f"Warning: low confidence, consider manual verification.")

    if model_path is None:
        model_path, model_type, metadata = find_best_model(
            exercise=exercise,
            model_type=args.model_type,
            models_base_dir=args.models_base_dir
        )
        print(f"Found best model for '{exercise}': {model_type} (loss={metadata['result']['best_test_loss']:.6f})")
        print(f"Model path: {model_path}")
    else:
        if model_type is None:
            parser.error("--model_type is required when using --model_path")

    print("Loading autoencoder...")
    model = load_model(model_path, model_type, device, metadata)

    os.makedirs(args.output_dir, exist_ok=True)
    normalize = not args.no_normalize

    if args.input_type == 'json':
        print(f"Processing JSON: {args.input}")
        errors, _ = process_json_file(args.input, model, device, args.seq_len, normalize)
    else:
        print(f"Processing video: {args.input}")
        errors, _ = process_video_file(args.input, model, device, args.seq_len, normalize, args.yolo_model)

    valid_mask = ~np.isnan(errors)
    errors_clean = errors[valid_mask]
    if len(errors_clean) == 0:
        print("No valid errors computed.")
        return

    threshold = args.threshold or np.percentile(errors_clean, args.threshold_percentile)
    print(f"Threshold: {threshold:.6f}")

    anomaly_frames = np.where(errors > threshold)[0]
    print(f"Anomalous frames: {len(anomaly_frames)} / {len(errors)}")

    results = {
        'threshold': float(threshold),
        'anomaly_frames': anomaly_frames.tolist(),
        'errors': errors.tolist(),
        'model_info': {
            'path': model_path,
            'type': model_type,
            'exercise': exercise,
            'best_test_loss': metadata['result']['best_test_loss'] if metadata else None
        }
    }
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    plot_path = os.path.join(args.output_dir, 'error_plot.png')
    plot_errors(errors, threshold, plot_path)

    print("Inference completed.")


if __name__ == '__main__':
    main()