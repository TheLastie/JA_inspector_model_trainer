import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from collections import defaultdict
import random

# -------------------- Конфигурация суставов (на основе разметки) --------------------
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

# -------------------- Загрузка данных из JSON --------------------
def load_angles_from_json(file_path, min_confidence=None):
    """Аналогично функции из train_rae_per_json.py"""
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
    """Перевод из радиан [0, π] в [-1, 1]"""
    return (angles / np.pi) * 2.0 - 1.0

def denormalize_angles(norm_angles):
    """Обратное преобразование [-1, 1] -> [0, π]"""
    return (norm_angles + 1.0) / 2.0 * np.pi

# -------------------- Генерация синтетических ошибок --------------------
def generate_error_sequence(correct_seq, error_prob=0.5, max_deviation_norm=0.4):
    """
    correct_seq: np.array (seq_len, num_joints) нормализованные углы [-1,1]
    Возвращает:
        noisy_seq: np.array той же формы
        labels: список кортежей (joint_name, error_class, deviation_norm)
                error_class: 0 - нет ошибки, 1 - меньше нормы, 2 - больше нормы
    """
    seq = correct_seq.copy()
    seq_len, num_joints = seq.shape
    labels = {name: (0, 0.0) for name in JOINT_NAMES}

    if np.random.rand() > error_prob:
        return seq, labels  # чистая последовательность

    # Выбираем один или несколько суставов для искажения
    num_errors = np.random.randint(1, 3)  # от 1 до 2 ошибок одновременно
    chosen_joints = np.random.choice(num_joints, size=num_errors, replace=False)

    for joint_idx in chosen_joints:
        joint_name = INDEX_TO_NAME[joint_idx]
        error_class = np.random.choice([1, 2])  # 1: меньше, 2: больше
        deviation = np.random.uniform(0.1, max_deviation_norm)
        if error_class == 1:
            deviation = -deviation

        # Тип искажения: постоянное или временное
        if np.random.rand() > 0.5:
            # Постоянное смещение на всей последовательности
            seq[:, joint_idx] += deviation
        else:
            # Временное: в середине движения (от 20% до 80% длины)
            start = int(0.2 * seq_len)
            end = int(0.8 * seq_len)
            seq[start:end, joint_idx] += deviation

        # Обрезаем значения
        seq[:, joint_idx] = np.clip(seq[:, joint_idx], -1.0, 1.0)
        labels[joint_name] = (error_class, deviation)

    return seq, labels

# -------------------- Датасет с синтетическими ошибками --------------------
class ErrorDetectionDataset(Dataset):
    def __init__(self, angles_array, boundaries, seq_len=30, samples_per_video=50,
                 error_prob=0.5, max_deviation=0.4):
        """
        angles_array: все кадры (total_frames, num_joints)
        boundaries: границы видео
        seq_len: длина окна
        samples_per_video: сколько синтетических примеров сгенерировать из каждого видео
        """
        self.seq_len = seq_len
        self.error_prob = error_prob
        self.max_deviation = max_deviation

        # Нормализуем все углы
        self.data = normalize_angles(angles_array)

        # Собираем все возможные стартовые индексы для окон (только внутри видео)
        self.valid_starts = []
        for start, end in boundaries:
            if end - start >= seq_len:
                for i in range(start, end - seq_len + 1):
                    self.valid_starts.append(i)

        # Для увеличения разнообразия будем генерировать несколько примеров из одного окна
        self.samples_per_window = max(1, samples_per_video // max(1, len(self.valid_starts)))

    def __len__(self):
        return len(self.valid_starts) * self.samples_per_window

    def __getitem__(self, idx):
        window_idx = idx // self.samples_per_window
        start = self.valid_starts[window_idx]
        correct_seq = self.data[start:start + self.seq_len]  # (seq_len, num_joints)

        noisy_seq, labels_dict = generate_error_sequence(
            correct_seq,
            error_prob=self.error_prob,
            max_deviation_norm=self.max_deviation
        )

        # Преобразуем словарь меток в тензоры
        class_labels = torch.zeros(NUM_JOINTS, dtype=torch.long)
        deviation_values = torch.zeros(NUM_JOINTS, dtype=torch.float32)
        for i, name in enumerate(JOINT_NAMES):
            cls, dev = labels_dict[name]
            class_labels[i] = cls
            deviation_values[i] = dev

        return torch.FloatTensor(noisy_seq), class_labels, deviation_values

# -------------------- Модель RAE (только энкодер) --------------------
class RAEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, latent_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        latent = self.fc(hidden[-1])
        return latent

# -------------------- Полная модель анализатора --------------------
class ExerciseAnalyzer(nn.Module):
    def __init__(self, encoder, latent_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.latent_size = latent_size

        self.shared_fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Классификационные головы для каждого сустава
        self.class_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, 3)  # 0,1,2
            ) for name in JOINT_NAMES
        })

        # Регрессионные головы для величины отклонения
        self.reg_heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, 1) for name in JOINT_NAMES
        })

    def forward(self, x):
        latent = self.encoder(x)
        shared = self.shared_fc(latent)

        class_logits = {}
        deviations = {}
        for name in JOINT_NAMES:
            class_logits[name] = self.class_heads[name](shared)
            deviations[name] = self.reg_heads[name](shared)

        return class_logits, deviations

# -------------------- Загрузка предобученного RAE --------------------
def load_pretrained_rae(model_path, input_size, device, args):
    """
    Загружает веса предобученного RAE напрямую в RAEncoder.
    Возвращает (encoder, latent_size).
    """
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

    dropout = getattr(args, 'dropout', 0.0)

    print(f"  [Загрузка RAE] hidden={hidden_size}, latent={latent_size}, layers={num_layers}")

    encoder = RAEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # Переносим веса энкодерной части
    encoder_state = {}
    for key, value in state_dict.items():
        if key.startswith('encoder_lstm.'):
            new_key = key.replace('encoder_lstm.', 'lstm.')
            encoder_state[new_key] = value
        elif key.startswith('encoder_fc.'):
            new_key = key.replace('encoder_fc.', 'fc.')
            encoder_state[new_key] = value

    encoder.load_state_dict(encoder_state)
    return encoder, latent_size
# -------------------- Обучение анализатора для одного упражнения --------------------
def train_analyzer_for_exercise(json_path, rae_model_path, output_base, args, ex_params):
    print(f"\n=== Обучение анализатора для: {os.path.basename(json_path)} ===")

    # Загрузка данных
    angles, boundaries = load_angles_from_json(json_path, min_confidence=ex_params.get('min_conf', None))
    if len(angles) == 0:
        print("  Нет данных. Пропускаем.")
        return

    input_size = angles.shape[1]
    print(f"  Всего кадров: {len(angles)}, признаков: {input_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем предобученный энкодер
        # Загружаем предобученный энкодер
    encoder, latent_size = load_pretrained_rae(rae_model_path, input_size, device, args)
    # Замораживаем энкодер (опционально)
    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    # Создаём датасет с синтетическими ошибками
    dataset = ErrorDetectionDataset(
        angles, boundaries,
        seq_len=ex_params['seq_len'],
        samples_per_video=args.samples_per_video,
        error_prob=args.error_prob,
        max_deviation=args.max_deviation
    )
    # Разделение на train/val (90/10)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Создаём модель анализатора
    analyzer = ExerciseAnalyzer(
        encoder,
        latent_size=latent_size,
        hidden_size=args.analyzer_hidden,
        dropout=args.dropout
    ).to(device)

    # Оптимизатор (только для голов и, возможно, энкодера)
    # В коде (после создания оптимизатора) можно задать:
optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': args.lr * 0.1},
    {'params': analyzer.shared_fc.parameters(), 'lr': args.lr},
    {'params': analyzer.class_heads.parameters(), 'lr': args.lr},
    {'params': analyzer.reg_heads.parameters(), 'lr': args.lr}
], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Папка для сохранения
    ex_name = os.path.splitext(os.path.basename(json_path))[0]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base, ex_name, f"analyzer_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"  Обучающих примеров: {train_size}, валидационных: {val_size}")
    print(f"  Результаты будут сохранены в: {run_dir}")

    # Цикл обучения
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        analyzer.train()
        total_loss = 0.0
        for batch_seq, class_labels, dev_values in train_loader:
            batch_seq = batch_seq.to(device)
            class_labels = class_labels.to(device)
            dev_values = dev_values.to(device)

            optimizer.zero_grad()
            class_logits, deviations = analyzer(batch_seq)

            loss = 0.0
            for i, name in enumerate(JOINT_NAMES):
                # Кросс-энтропия для классификации
                loss += F.cross_entropy(class_logits[name], class_labels[:, i])
                # MSE для регрессии отклонения (только для кадров с ошибкой)
                mask = (class_labels[:, i] != 0).float().unsqueeze(1)
                if mask.sum() > 0:
                    reg_loss = F.mse_loss(deviations[name] * mask, dev_values[:, i:i+1] * mask)
                    loss += args.reg_weight * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(analyzer.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch_seq.size(0)

        avg_train_loss = total_loss / train_size
        train_losses.append(avg_train_loss)

        # Валидация
        analyzer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_seq, class_labels, dev_values in val_loader:
                batch_seq = batch_seq.to(device)
                class_labels = class_labels.to(device)
                dev_values = dev_values.to(device)
                class_logits, deviations = analyzer(batch_seq)

                loss = 0.0
                for i, name in enumerate(JOINT_NAMES):
                    loss += F.cross_entropy(class_logits[name], class_labels[:, i])
                    mask = (class_labels[:, i] != 0).float().unsqueeze(1)
                    if mask.sum() > 0:
                        reg_loss = F.mse_loss(deviations[name] * mask, dev_values[:, i:i+1] * mask)
                        loss += args.reg_weight * reg_loss
                val_loss += loss.item() * batch_seq.size(0)
        avg_val_loss = val_loss / val_size
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(analyzer.state_dict(), os.path.join(run_dir, 'best_analyzer.pth'))
        else:
            patience_counter += 1

        if epoch % args.log_interval == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"  Ранняя остановка на эпохе {epoch}")
            break

    # График обучения
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Analyzer training: {ex_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'loss_plot.png'))
    plt.close()

    # Сохранение лога
    with open(os.path.join(run_dir, 'training_log.txt'), 'w') as f:
        f.write(f"JSON file: {json_path}\n")
        f.write(f"RAE model: {rae_model_path}\n")
        f.write(f"Parameters:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nBest Val Loss: {best_val_loss:.4f}\n")

    print(f"  Обучение завершено. Лучшая вал. ошибка: {best_val_loss:.4f}")
    return run_dir

# -------------------- Главная функция --------------------
def main():
    parser = argparse.ArgumentParser(description='Train Exercise Analyzer with synthetic errors')
    parser.add_argument('--data_dir', type=str, default='datasets', help='Папка с JSON файлами')
    parser.add_argument('--rae_models_dir', type=str, default='models', help='Папка с обученными RAE')
    parser.add_argument('--output_base', type=str, default='analyzers', help='Папка для сохранения анализаторов')
    parser.add_argument('--config', type=str, default='exercise_config.json', help='JSON с настройками упражнений')

    # Параметры модели (будут переопределены при загрузке RAE)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--autoregressive', action='store_true', default=True)

    # Параметры генерации синтетических ошибок
    parser.add_argument('--seq_len', type=int, default=30, help='Длина окна (должна совпадать с RAE)')
    parser.add_argument('--samples_per_video', type=int, default=100, help='Число примеров на одно видео')
    parser.add_argument('--error_prob', type=float, default=0.6, help='Вероятность внесения ошибки')
    parser.add_argument('--max_deviation', type=float, default=0.5, help='Макс. отклонение в нормализованных единицах')

    # Параметры обучения анализатора
    parser.add_argument('--analyzer_hidden', type=int, default=128, help='Размер скрытого слоя в головах')
    parser.add_argument('--freeze_encoder', action='store_true', default=True, help='Заморозить энкодер')
    parser.add_argument('--reg_weight', type=float, default=0.5, help='Вес регрессионной части loss')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)

    # --- Загрузка или создание конфигурационного файла ---
    exercise_config = {"default": {}}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    exercise_config = json.loads(content)
                else:
                    print(f"⚠️ Файл {args.config} пуст. Будет создан шаблон.")
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка чтения {args.config}: {e}. Будет создан новый шаблон.")
            backup_name = args.config + '.backup'
            os.rename(args.config, backup_name)
            print(f"   Битый файл переименован в {backup_name}")
    else:
        print(f"ℹ️ Файл {args.config} не найден. Будет создан шаблон.")

    if not exercise_config or "default" not in exercise_config:
        exercise_config = {
            "default": {
                "seq_len": args.seq_len,
                "hidden_size": args.hidden_size,
                "latent_size": args.latent_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "autoregressive": args.autoregressive,
                "min_conf": None
            }
        }
        with open(args.config, 'w', encoding='utf-8') as f:
            json.dump(exercise_config, f, indent=2, ensure_ascii=False)
        print(f"✅ Создан шаблон конфигурации: {args.config}")

    default_params = exercise_config.get("default", {})

    json_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    if not json_files:
        print(f"В папке {args.data_dir} нет JSON файлов.")
        return

    print(f"Найдено JSON файлов: {len(json_files)}")
    if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
            print(f"Using GPU {args.gpu_id}")
    for json_file in json_files:
        ex_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(args.data_dir, json_file)

        # Ищем соответствующую RAE модель (последний запуск)
        rae_model_dir = os.path.join(args.rae_models_dir, ex_name)
        if not os.path.exists(rae_model_dir):
            print(f"  Для {ex_name} нет обученной RAE в {rae_model_dir}, пропускаем.")
            continue

        runs = [d for d in os.listdir(rae_model_dir) if d.startswith('run_')]
        if not runs:
            print(f"  Для {ex_name} нет запусков RAE, пропускаем.")
            continue
        latest_run = sorted(runs)[-1]
        rae_model_path = os.path.join(rae_model_dir, latest_run, 'best_model.pth')
        if not os.path.exists(rae_model_path):
            print(f"  Не найден best_model.pth в {rae_model_dir}/{latest_run}, пропускаем.")
            continue

        ex_params = {**default_params, **exercise_config.get(ex_name, {})}
        ex_params.setdefault('seq_len', args.seq_len)
        ex_params.setdefault('min_conf', None)
        
        try:
            train_analyzer_for_exercise(json_path, rae_model_path, args.output_base, args, ex_params)
        except Exception as e:
            print(f"  Ошибка при обработке {json_file}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()