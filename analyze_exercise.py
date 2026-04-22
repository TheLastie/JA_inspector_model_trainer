import json
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from collections import defaultdict

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

# -------------------- Нормализация --------------------
def normalize_angles(angles):
    return (angles / np.pi) * 2.0 - 1.0

def denormalize_angles(norm_angles):
    return (norm_angles + 1.0) / 2.0 * np.pi

# -------------------- Загрузка данных из JSON --------------------
def load_angles_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    all_angles = []
    video_boundaries = []
    current_idx = 0

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "vids_list":
                continue
            if isinstance(value, dict):
                sorted_frames = sorted(value.items(), key=lambda x: int(x[0]))
                video_angles = [fd['angles'] for _, fd in sorted_frames if 'angles' in fd]
            elif isinstance(value, list):
                video_angles = [item['angles'] for item in value if 'angles' in item]
            else:
                continue

            if video_angles:
                video_boundaries.append((current_idx, current_idx + len(video_angles)))
                all_angles.extend(video_angles)
                current_idx += len(video_angles)

    elif isinstance(data, list):
        video_angles = [item['angles'] for item in data if 'angles' in item]
        if video_angles:
            video_boundaries.append((0, len(video_angles)))
            all_angles = video_angles
    else:
        raise ValueError(f"Неподдерживаемый тип данных в JSON: {type(data)}")

    if not all_angles:
        return np.array([], dtype=np.float32), []
    return np.array(all_angles, dtype=np.float32), video_boundaries

# -------------------- Модель RAE (такая же, как в обучении) --------------------
class RecurrentAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, latent_size=64, num_layers=2, dropout=0.2, autoregressive=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.autoregressive = autoregressive

        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder_output = nn.Linear(hidden_size, input_size)

        if self.autoregressive:
            self.start_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.output_to_hidden = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(hidden[-1])

        # Decode
        decoder_hidden = self.decoder_fc(latent)
        h_t = decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_t = torch.zeros_like(h_t)

        if self.autoregressive:
            decoder_input = self.start_token.repeat(batch_size, 1, 1)
            outputs = []
            for t in range(seq_len):
                out, (h_t, c_t) = self.decoder_lstm(decoder_input, (h_t, c_t))
                pred = self.decoder_output(out)
                outputs.append(pred)
                decoder_input = self.output_to_hidden(pred)
            reconstructed = torch.cat(outputs, dim=1)
        else:
            decoder_input = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            out, _ = self.decoder_lstm(decoder_input, (h_t, c_t))
            reconstructed = self.decoder_output(out)

        return reconstructed, latent

# -------------------- Загрузка модели с автоопределением архитектуры --------------------
def load_rae_model(model_path, input_size, device):
    state_dict = torch.load(model_path, map_location=device)

    # Определяем hidden_size
    key_ih = 'encoder_lstm.weight_ih_l0'
    if key_ih not in state_dict:
        raise ValueError(f"Чекпоинт не содержит ключ {key_ih}")
    hidden_size = state_dict[key_ih].shape[0] // 4

    # Определяем latent_size
    key_fc = 'encoder_fc.weight'
    if key_fc not in state_dict:
        raise ValueError(f"Чекпоинт не содержит ключ {key_fc}")
    latent_size = state_dict[key_fc].shape[0]

    # Определяем num_layers
    num_layers = 0
    for key in state_dict.keys():
        if key.startswith('encoder_lstm.weight_ih_l'):
            layer_idx = int(key.split('_l')[-1])
            num_layers = max(num_layers, layer_idx + 1)
    if num_layers == 0:
        num_layers = 1

    # Проверяем, авторегрессионная ли модель (есть ли start_token)
    autoregressive = 'start_token' in state_dict

    model = RecurrentAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        dropout=0.0,  # для инференса не важно
        autoregressive=autoregressive
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -------------------- Вычисление порогов ошибок --------------------
def compute_error_thresholds(model, dataloader, device, percentile=95):
    """Вычисляет перцентиль ошибки восстановления для каждого сустава на тренировочных данных."""
    all_errors = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            errors = (reconstructed - batch).cpu().numpy()  # (batch, seq_len, 11)
            # Собираем ошибки по каждому суставу (все кадры, все батчи)
            for i, name in enumerate(JOINT_NAMES):
                all_errors[name].extend(errors[:, :, i].flatten().tolist())
    thresholds = {}
    for name in JOINT_NAMES:
        thresholds[name] = np.percentile(np.abs(all_errors[name]), percentile)
    return thresholds

# -------------------- Анализ одной последовательности --------------------
def analyze_sequence(model, sequence, thresholds, device, significance=1.5):
    """
    sequence: np.array (seq_len, 11) в радианах.
    Возвращает словарь с результатами для каждого сустава.
    """
    norm_seq = normalize_angles(sequence)
    input_tensor = torch.FloatTensor(norm_seq).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed, _ = model(input_tensor)
        reconstructed = reconstructed.cpu().numpy()[0]  # (seq_len, 11)

    errors = reconstructed - norm_seq  # положительное => модель считает, что угол должен быть больше

    results = {}
    for i, name in enumerate(JOINT_NAMES):
        joint_errors = errors[:, i]
        mean_abs_error = np.mean(np.abs(joint_errors))
        mean_error = np.mean(joint_errors)  # знак указывает направление

        threshold = thresholds[name] * significance
        is_anomaly = mean_abs_error > threshold

        if is_anomaly:
            direction = "меньше нормы" if mean_error > 0 else "больше нормы"
            # Переводим отклонение в градусы (среднее по времени)
            deviation_deg = denormalize_angles(np.abs(mean_error)) * 180 / np.pi
        else:
            direction = "норма"
            deviation_deg = 0.0

        results[name] = {
            'is_anomaly': is_anomaly,
            'direction': direction,
            'mean_abs_error_norm': mean_abs_error,
            'threshold_norm': threshold,
            'deviation_deg': deviation_deg
        }
    return results

# -------------------- Главная функция --------------------
def main():
    parser = argparse.ArgumentParser(description='Детектор ошибок на основе RAE')
    parser.add_argument('--json_path', type=str, required=True, help='Путь к JSON-файлу с данными упражнения')
    parser.add_argument('--rae_model', type=str, required=True, help='Путь к best_model.pth')
    parser.add_argument('--seq_len', type=int, default=30, help='Длина окна (должна совпадать с обучением)')
    parser.add_argument('--percentile', type=int, default=95, help='Перцентиль для порога ошибки')
    parser.add_argument('--significance', type=float, default=1.5, help='Множитель порога (1.5 = умеренная чувствительность)')
    parser.add_argument('--test_video_index', type=int, default=-1, help='Индекс видео для теста (-1 = последнее)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем данные
    print(f"Загрузка данных из {args.json_path}...")
    angles, boundaries = load_angles_from_json(args.json_path)
    if len(angles) == 0:
        print("Нет данных.")
        return
    input_size = angles.shape[1]
    norm_angles = normalize_angles(angles)

    # Разделение на train/test по видео
    np.random.seed(42)
    video_indices = list(range(len(boundaries)))
    np.random.shuffle(video_indices)
    test_video_idx = video_indices[args.test_video_index] if args.test_video_index >= 0 else video_indices[-1]
    train_boundaries = [b for i, b in enumerate(boundaries) if i != test_video_idx]
    test_boundary = boundaries[test_video_idx]

    # Создаём датасеты
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data, seq_len):
            self.data = data
            self.seq_len = seq_len
        def __len__(self):
            return max(0, len(self.data) - self.seq_len)
        def __getitem__(self, idx):
            return torch.FloatTensor(self.data[idx:idx+self.seq_len])

    train_data = np.concatenate([norm_angles[s:e] for s, e in train_boundaries], axis=0)
    train_dataset = SimpleDataset(train_data, args.seq_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Загружаем модель
    print(f"Загрузка модели из {args.rae_model}...")
    model = load_rae_model(args.rae_model, input_size, device)

    # Вычисляем пороги
    print("Вычисление порогов ошибок по тренировочным данным...")
    thresholds = compute_error_thresholds(model, train_loader, device, percentile=args.percentile)

    # Анализируем тестовое видео
    test_start, test_end = test_boundary
    test_angles = angles[test_start:test_end]
    if len(test_angles) < args.seq_len:
        print(f"Тестовое видео слишком короткое (нужно минимум {args.seq_len} кадров).")
        return

    # Анализируем несколько окон из тестового видео (например, 3 окна: начало, середина, конец)
    windows = [
        test_angles[:args.seq_len],
        test_angles[len(test_angles)//2 - args.seq_len//2 : len(test_angles)//2 + args.seq_len//2],
        test_angles[-args.seq_len:]
    ]
    window_names = ["начало", "середина", "конец"]

    print(f"\n=== Анализ тестового видео (индекс {test_video_idx}) ===")
    for name, win in zip(window_names, windows):
        results = analyze_sequence(model, win, thresholds, device, significance=args.significance)
        print(f"\n--- Окно: {name} ---")
        anomalies = []
        for joint, res in results.items():
            if res['is_anomaly']:
                anomalies.append(f"{joint}: {res['direction']} (отклонение ~{res['deviation_deg']:.1f}°)")
        if anomalies:
            for a in anomalies:
                print(f"  ⚠️ {a}")
        else:
            print("  ✅ Все суставы в норме.")

    print("\nПороги ошибок (нормализованные):")
    for name in JOINT_NAMES:
        print(f"  {name}: {thresholds[name]:.4f}")

if __name__ == '__main__':
    main()