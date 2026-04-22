#!/usr/bin/env python3
"""
Расширенный скрипт обучения ExerciseAnalyzer с реалистичной генерацией ошибок,
подбором гиперпараметров через Optuna и сохранением результатов.
"""

import json
import os
import sys
import argparse
import uuid
import traceback
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
import pandas as pd

# -------------------- Конфигурация суставов --------------------
INDEX_TO_NAME = {
    0: 'left_shoulder',
    1: 'neck',
    2: 'hip_opening',
    3: 'spine',
    4: 'right_shoulder',
    5: 'right_elbow',
    6: 'right_hip',
    7: 'left_hip',
    8: 'left_knee',
    9: 'right_knee',
    10: 'left_elbow'
}
JOINT_NAMES = list(INDEX_TO_NAME.values())
NUM_JOINTS = len(JOINT_NAMES)

# -------------------- Типовые ошибки для упражнений --------------------
EXERCISE_ERRORS = {
    'lateral_raise': [
        {'name': 'body_swing', 'joints': ['spine', 'left_shoulder', 'right_shoulder'], 'strength': 0.2},
        {'name': 'elbow_bend', 'joints': ['left_elbow', 'right_elbow'], 'strength': 0.25},
        {'name': 'uneven_arms', 'joints': ['left_shoulder', 'right_shoulder'], 'strength': 0.2},
    ],
    'squat': [
        {'name': 'knees_in', 'joints': ['left_knee', 'right_knee'], 'strength': 0.3},
        {'name': 'back_round', 'joints': ['spine', 'hip_opening'], 'strength': 0.2},
        {'name': 'shallow_depth', 'joints': ['left_hip', 'right_hip', 'left_knee', 'right_knee'], 'strength': 0.3},
    ],
    'deadlift': [
        {'name': 'back_round', 'joints': ['spine', 'hip_opening'], 'strength': 0.3},
        {'name': 'shoulder_shrug', 'joints': ['left_shoulder', 'right_shoulder'], 'strength': 0.2},
        {'name': 'incomplete_lockout', 'joints': ['left_hip', 'right_hip'], 'strength': 0.2},
    ],
    'bench_press': [
        {'name': 'hip_raise', 'joints': ['left_hip', 'right_hip'], 'strength': 0.2},
        {'name': 'elbow_flare', 'joints': ['left_elbow', 'right_elbow'], 'strength': 0.3},
        {'name': 'uneven_lockout', 'joints': ['left_shoulder', 'right_shoulder'], 'strength': 0.2},
    ],
    'plank': [
        {'name': 'hip_sag', 'joints': ['left_hip', 'right_hip', 'spine'], 'strength': 0.25},
        {'name': 'hip_pike', 'joints': ['left_hip', 'right_hip', 'spine'], 'strength': 0.2},
        {'name': 'shoulder_protraction', 'joints': ['left_shoulder', 'right_shoulder'], 'strength': 0.15},
    ],
    'default': [
        {'name': 'random_joint_noise', 'joints': [], 'strength': 0.1},
    ]
}

# -------------------- Загрузка данных из JSON --------------------
def load_angles_from_json(file_path: str, min_confidence: Optional[float] = None):
    with open(file_path, 'r') as f:
        data = json.load(f)

    all_angles = []
    video_boundaries = []
    current_idx = 0

    def frame_passes(frame_data):
        if min_confidence is None:
            return True
        conf = frame_data.get('conf')
        if conf is None:
            return True
        if isinstance(conf, (list, tuple)):
            return all(c >= min_confidence for c in conf)
        else:
            return conf >= min_confidence

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "vids_list":
                continue
            if isinstance(value, dict):
                sorted_frames = sorted(value.items(), key=lambda x: int(x[0]))
                video_angles = [fd['angles'] for _, fd in sorted_frames if 'angles' in fd and frame_passes(fd)]
            elif isinstance(value, list):
                video_angles = [item['angles'] for item in value if 'angles' in item and frame_passes(item)]
            else:
                continue

            if video_angles:
                video_boundaries.append((current_idx, current_idx + len(video_angles)))
                all_angles.extend(video_angles)
                current_idx += len(video_angles)

    elif isinstance(data, list):
        video_angles = [item['angles'] for item in data if 'angles' in item and frame_passes(item)]
        if video_angles:
            video_boundaries.append((0, len(video_angles)))
            all_angles = video_angles
    else:
        raise ValueError(f"Неподдерживаемый тип данных в JSON: {type(data)}")

    if not all_angles:
        return np.array([], dtype=np.float32), []
    return np.array(all_angles, dtype=np.float32), video_boundaries

# -------------------- Нормализация --------------------
def normalize_angles(angles):
    return (angles / np.pi) * 2.0 - 1.0

def denormalize_angles(norm_angles):
    return (norm_angles + 1.0) / 2.0 * np.pi

# -------------------- Реалистичная генерация ошибок --------------------
def apply_realistic_error(angles_seq: np.ndarray, exercise_name: str, error_prob: float = 0.5) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
    """
    Применяет реалистичную ошибку к последовательности углов.
    Возвращает:
        изменённую последовательность (seq_len, num_joints),
        словарь с метками классов (0/1) для каждого сустава,
        словарь с величиной отклонения (в нормализованных единицах).
    """
    seq = angles_seq.copy()
    seq_len, num_joints = seq.shape
    labels_class = {name: 0 for name in JOINT_NAMES}
    labels_dev = {name: 0.0 for name in JOINT_NAMES}

    if np.random.rand() > error_prob:
        return seq, labels_class, labels_dev

    ex_name = exercise_name.lower().replace('.json', '')
    error_templates = EXERCISE_ERRORS.get(ex_name, EXERCISE_ERRORS['default'])
    template = np.random.choice(error_templates)

    affected_joints = template['joints']
    strength = template['strength'] * np.random.uniform(0.7, 1.3)

    if template['name'] == 'random_joint_noise' or not affected_joints:
        # Случайно выбираем 1-2 сустава
        affected_indices = np.random.choice(num_joints, size=np.random.randint(1, 3), replace=False)
        affected_joints = [INDEX_TO_NAME[i] for i in affected_indices]

    for joint_name in affected_joints:
        if joint_name not in JOINT_NAMES:
            continue
        idx = JOINT_NAMES.index(joint_name)

        # Направление ошибки (больше или меньше нормы)
        sign = np.random.choice([-1, 1])
        # Величина отклонения
        deviation = strength * np.random.uniform(0.5, 1.5)

        # Применяем искажение (может быть временным или постоянным)
        if np.random.rand() > 0.5:
            # Постоянное смещение на всей последовательности
            seq[:, idx] += sign * deviation
        else:
            # Временное (в середине движения)
            start = int(0.2 * seq_len)
            end = int(0.8 * seq_len)
            seq[start:end, idx] += sign * deviation

        # Обрезаем значения, чтобы оставаться в пределах нормированного диапазона
        seq[:, idx] = np.clip(seq[:, idx], -1.0, 1.0)

        labels_class[joint_name] = 1
        labels_dev[joint_name] = sign * deviation

    return seq, labels_class, labels_dev

# -------------------- Датасет --------------------
class RealisticErrorDataset(Dataset):
    def __init__(self, angles_array, boundaries, exercise_name, seq_len=30,
                 samples_per_video=100, error_prob=0.5):
        self.seq_len = seq_len
        self.exercise_name = exercise_name
        self.error_prob = error_prob
        self.data = normalize_angles(angles_array)

        self.valid_starts = []
        for start, end in boundaries:
            if end - start >= seq_len:
                for i in range(start, end - seq_len + 1):
                    self.valid_starts.append(i)

        self.samples_per_window = max(1, samples_per_video // max(1, len(self.valid_starts)))

    def __len__(self):
        return len(self.valid_starts) * self.samples_per_window

    def __getitem__(self, idx):
        window_idx = idx // self.samples_per_window
        start = self.valid_starts[window_idx]
        correct_seq = self.data[start:start + self.seq_len]

        noisy_seq, labels_class, labels_dev = apply_realistic_error(
            correct_seq, self.exercise_name, self.error_prob
        )

        # Преобразуем в тензоры
        class_tensor = torch.zeros(NUM_JOINTS, dtype=torch.long)
        dev_tensor = torch.zeros(NUM_JOINTS, dtype=torch.float32)
        for i, name in enumerate(JOINT_NAMES):
            class_tensor[i] = labels_class[name]
            dev_tensor[i] = labels_dev[name]

        return torch.FloatTensor(noisy_seq), class_tensor, dev_tensor

# -------------------- Энкодер (RAE) --------------------
class RAEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, latent_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# -------------------- Анализатор --------------------
class ExerciseAnalyzer(nn.Module):
    def __init__(self, encoder, latent_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.shared_fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Бинарные классификаторы для каждого сустава
        self.class_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)  # выход 1 для бинарной классификации
            ) for name in JOINT_NAMES
        })
        # Регрессоры величины отклонения
        self.reg_heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, 1) for name in JOINT_NAMES
        })

    def forward(self, x):
        latent = self.encoder(x)
        shared = self.shared_fc(latent)
        class_logits = {name: self.class_heads[name](shared) for name in JOINT_NAMES}
        deviations = {name: self.reg_heads[name](shared) for name in JOINT_NAMES}
        return class_logits, deviations

# -------------------- Загрузка предобученного RAE --------------------
def load_pretrained_rae(model_path: str, input_size: int, device: torch.device):
    state_dict = torch.load(model_path, map_location=device)

    key_ih = 'encoder_lstm.weight_ih_l0'
    if key_ih not in state_dict:
        raise ValueError(f"Чекпоинт не содержит ключ {key_ih}")
    hidden_size = state_dict[key_ih].shape[0] // 4

    key_fc = 'encoder_fc.weight'
    latent_size = state_dict[key_fc].shape[0]

    num_layers = 0
    for key in state_dict.keys():
        if key.startswith('encoder_lstm.weight_ih_l'):
            layer_idx = int(key.split('_l')[-1])
            num_layers = max(num_layers, layer_idx + 1)
    if num_layers == 0:
        num_layers = 1

    encoder = RAEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        dropout=0.0  # не важно для инференса
    ).to(device)

    encoder_state = {}
    for key, value in state_dict.items():
        if key.startswith('encoder_lstm.'):
            encoder_state[key.replace('encoder_lstm.', 'lstm.')] = value
        elif key.startswith('encoder_fc.'):
            encoder_state[key.replace('encoder_fc.', 'fc.')] = value

    encoder.load_state_dict(encoder_state)
    return encoder, latent_size

# -------------------- Обучение с заданными параметрами --------------------
def train_analyzer(
    json_path: str,
    rae_model_path: str,
    output_dir: str,
    params: Dict[str, Any],
    device: torch.device
) -> float:
    """
    Обучает анализатор с заданными параметрами.
    Возвращает лучший validation loss.
    """
    exercise_name = os.path.splitext(os.path.basename(json_path))[0]
    print(f"\n=== Training analyzer for {exercise_name} ===")

    # Загрузка данных
    angles, boundaries = load_angles_from_json(json_path)
    if len(angles) == 0:
        raise ValueError("No data in JSON")
    input_size = angles.shape[1]

    # Загрузка энкодера
    encoder, latent_size = load_pretrained_rae(rae_model_path, input_size, device)

    # Заморозка энкодера
    if params.get('freeze_encoder', True):
        for param in encoder.parameters():
            param.requires_grad = False

    # Создание датасета
    dataset = RealisticErrorDataset(
        angles, boundaries,
        exercise_name=exercise_name,
        seq_len=params['seq_len'],
        samples_per_video=params['samples_per_video'],
        error_prob=params['error_prob']
    )

    # Разделение
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # Модель
    analyzer = ExerciseAnalyzer(
        encoder,
        latent_size=latent_size,
        hidden_size=params['analyzer_hidden'],
        dropout=params['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, analyzer.parameters()),
        lr=params['lr']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=params['scheduler_patience']
    )

    # Функция потерь
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, params['epochs'] + 1):
        analyzer.train()
        train_loss = 0.0
        for batch_seq, class_labels, dev_values in train_loader:
            batch_seq = batch_seq.to(device)
            class_labels = class_labels.to(device).float()  # (batch, NUM_JOINTS)
            dev_values = dev_values.to(device)

            optimizer.zero_grad()
            class_logits, deviations = analyzer(batch_seq)

            loss = 0.0
            for i, name in enumerate(JOINT_NAMES):
                # Бинарная кросс-энтропия
                loss += bce_loss(class_logits[name].squeeze(1), class_labels[:, i])

                # Регрессия отклонения только для кадров с ошибкой
                mask = (class_labels[:, i] == 1).float().unsqueeze(1)
                if mask.sum() > 0:
                    reg_loss = F.mse_loss(deviations[name] * mask, dev_values[:, i:i+1] * mask)
                    loss += params['reg_weight'] * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(analyzer.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_seq.size(0)

        avg_train_loss = train_loss / train_size

        # Валидация
        analyzer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_seq, class_labels, dev_values in val_loader:
                batch_seq = batch_seq.to(device)
                class_labels = class_labels.to(device).float()
                dev_values = dev_values.to(device)
                class_logits, deviations = analyzer(batch_seq)

                loss = 0.0
                for i, name in enumerate(JOINT_NAMES):
                    loss += bce_loss(class_logits[name].squeeze(1), class_labels[:, i])
                    mask = (class_labels[:, i] == 1).float().unsqueeze(1)
                    if mask.sum() > 0:
                        reg_loss = F.mse_loss(deviations[name] * mask, dev_values[:, i:i+1] * mask)
                        loss += params['reg_weight'] * reg_loss
                val_loss += loss.item() * batch_seq.size(0)
        avg_val_loss = val_loss / val_size

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = analyzer.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= params['early_stop_patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Сохранение лучшей модели
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
    save_dir = os.path.join(output_dir, exercise_name, f"analyzer_{run_id}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, os.path.join(save_dir, 'best_analyzer.pth'))

    # Сохранение параметров и результата
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
        f.write(f"Best Val Loss: {best_val_loss:.6f}\n")

    return best_val_loss

# -------------------- Обёртка для Optuna --------------------
def objective(trial: Trial, json_path: str, rae_model_path: str, output_dir: str, device: torch.device) -> float:
    params = {
        'seq_len': 30,  # фиксировано, т.к. RAE обучена с этим
        'samples_per_video': trial.suggest_int('samples_per_video', 100, 500, step=100),
        'error_prob': trial.suggest_float('error_prob', 0.4, 0.8),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'analyzer_hidden': trial.suggest_categorical('analyzer_hidden', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'reg_weight': trial.suggest_float('reg_weight', 0.1, 1.0),
        'epochs': 50,  # для подбора достаточно
        'early_stop_patience': 10,
        'scheduler_patience': 5,
        'freeze_encoder': True,
    }
    return train_analyzer(json_path, rae_model_path, output_dir, params, device)

# -------------------- Главная функция --------------------
def main():
    parser = argparse.ArgumentParser(description='Advanced Exercise Analyzer Training with Hyperparameter Tuning')
    parser.add_argument('--data_dir', default='datasets')
    parser.add_argument('--rae_models_dir', default='models')
    parser.add_argument('--output_dir', default='analyzers_advanced')
    parser.add_argument('--exercise', help='Specific exercise to train (default: all)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials per exercise')
    parser.add_argument('--study_name', default='analyzer_study')
    parser.add_argument('--storage', default='sqlite:///analyzer_study.db', help='Optuna storage URL')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Определяем список упражнений
    if args.exercise:
        exercises = [args.exercise]
    else:
        exercises = [f.replace('.json', '') for f in os.listdir(args.data_dir) if f.endswith('.json')]

    all_results = []

    for ex in exercises:
        json_path = os.path.join(args.data_dir, f"{ex}.json")
        if not os.path.exists(json_path):
            print(f"JSON not found for {ex}, skipping.")
            continue

        # Ищем RAE модель
        rae_dir = os.path.join(args.rae_models_dir, ex)
        if not os.path.exists(rae_dir):
            print(f"No RAE models for {ex}, skipping.")
            continue
        runs = [d for d in os.listdir(rae_dir) if d.startswith('run_')]
        if not runs:
            print(f"No RAE runs for {ex}, skipping.")
            continue
        latest_run = sorted(runs)[-1]
        rae_model_path = os.path.join(rae_dir, latest_run, 'best_model.pth')
        if not os.path.exists(rae_model_path):
            print(f"RAE model not found for {ex}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Training analyzer for {ex} with RAE from {latest_run}")
        print('='*60)

        # Создаём отдельное study для упражнения
        study = optuna.create_study(
            study_name=f"{args.study_name}_{ex}",
            storage=args.storage,
            sampler=TPESampler(seed=42),
            direction='minimize',
            load_if_exists=True
        )

        def obj_wrapper(trial):
            return objective(trial, json_path, rae_model_path, args.output_dir, device)

        study.optimize(obj_wrapper, n_trials=args.n_trials, show_progress_bar=True)

        # Сохраняем результаты
        best_trial = study.best_trial
        result = {
            'exercise': ex,
            'best_val_loss': best_trial.value,
            'best_params': best_trial.params,
        }
        all_results.append(result)

        # Дополнительно обучаем с лучшими параметрами больше эпох
        print(f"\nTraining final model for {ex} with best params...")
        best_params = best_trial.params.copy()
        best_params.update({
            'seq_len': 30,
            'epochs': 150,
            'early_stop_patience': 30,
            'scheduler_patience': 10,
            'freeze_encoder': True,
        })
        final_loss = train_analyzer(json_path, rae_model_path, args.output_dir, best_params, device)
        result['final_val_loss'] = final_loss

    # Итоговая таблица
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'tuning_results.csv'), index=False)
    print("\n" + "="*60)
    print("All exercises processed. Summary:")
    print(df.to_string())

if __name__ == '__main__':
    main()