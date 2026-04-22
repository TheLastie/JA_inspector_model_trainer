import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# --- Загрузка данных ---
def load_angles_from_json(file_path, min_confidence=None):
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

# --- Нормализация ---
def normalize_angles(angles):
    return (angles / np.pi) * 2.0 - 1.0

class AngleSequenceDataset(Dataset):
    def __init__(self, angles_array, seq_len=30, normalize=True):
        self.seq_len = seq_len
        self.data = angles_array.copy()
        if normalize and len(self.data) > 0:
            self.data = normalize_angles(self.data)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx + self.seq_len])

def split_by_videos(angles_array, boundaries, test_ratio=0.1, seq_len=30):
    np.random.seed(42)
    video_indices = list(range(len(boundaries)))
    np.random.shuffle(video_indices)

    total_frames = len(angles_array)
    test_frames_needed = int(total_frames * test_ratio)

    test_frames_collected = 0
    test_videos = []
    train_videos = []

    for idx in video_indices:
        start, end = boundaries[idx]
        length = end - start
        if test_frames_collected < test_frames_needed:
            test_videos.append((start, end))
            test_frames_collected += length
        else:
            train_videos.append((start, end))

    def collect_frames(video_list):
        frames = []
        for start, end in video_list:
            frames.append(angles_array[start:end])
        if frames:
            return np.concatenate(frames, axis=0)
        return np.array([], dtype=np.float32)

    train_data = collect_frames(train_videos)
    test_data = collect_frames(test_videos)

    train_dataset = AngleSequenceDataset(train_data, seq_len) if len(train_data) >= seq_len else None
    test_dataset = AngleSequenceDataset(test_data, seq_len) if len(test_data) >= seq_len else None

    return train_dataset, test_dataset

# --- Модель RAE ---
class RecurrentAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, latent_size=64, num_layers=2,
                 dropout=0.2, autoregressive=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.autoregressive = autoregressive

        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.decoder_output = nn.Linear(hidden_size, input_size)

        if self.autoregressive:
            self.start_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.output_to_hidden = nn.Linear(input_size, hidden_size)

    def forward(self, x, teacher_forcing_ratio=0.0):
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
                if teacher_forcing_ratio > 0 and torch.rand(1).item() < teacher_forcing_ratio:
                    decoder_input = self.output_to_hidden(x[:, t:t+1, :])
                else:
                    decoder_input = self.output_to_hidden(pred)
            reconstructed = torch.cat(outputs, dim=1)
        else:
            decoder_input = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            out, _ = self.decoder_lstm(decoder_input, (h_t, c_t))
            reconstructed = self.decoder_output(out)

        return reconstructed, latent

# --- Обучение одной модели ---
def train_model_for_json(json_path, output_base, args):
    print(f"\n=== Обработка файла: {os.path.basename(json_path)} ===")

    angles, boundaries = load_angles_from_json(json_path, min_confidence=args.min_conf)
    if len(angles) == 0:
        print("  Нет данных. Пропускаем.")
        return None, None

    input_size = angles.shape[1]
    print(f"  Всего кадров: {len(angles)}, признаков: {input_size}")

    train_dataset, test_dataset = split_by_videos(angles, boundaries,
                                                  test_ratio=args.test_ratio,
                                                  seq_len=args.seq_len)
    if train_dataset is None:
        print("  Недостаточно данных для train. Пропускаем.")
        return None, None
    if test_dataset is None:
        split_idx = int(len(train_dataset) * 0.9)
        test_dataset = torch.utils.data.Subset(train_dataset, range(split_idx, len(train_dataset)))
        train_dataset = torch.utils.data.Subset(train_dataset, range(0, split_idx))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Установка GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"  Using GPU {args.gpu_id}")
    else:
        device = torch.device('cpu')

    model = RecurrentAutoencoder(
        input_size=input_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        autoregressive=args.autoregressive
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model_name = os.path.splitext(os.path.basename(json_path))[0]
    model_dir = os.path.join(output_base, model_name)
    os.makedirs(model_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(model_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    tf_ratio = args.teacher_forcing

    print(f"  Результаты: {run_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch, teacher_forcing_ratio=tf_ratio)
            loss = criterion(reconstructed, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * batch.size(0)

        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                reconstructed, _ = model(batch, teacher_forcing_ratio=0.0)
                loss = criterion(reconstructed, batch)
                total_test_loss += loss.item() * batch.size(0)
        avg_test_loss = total_test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)

        if args.scheduler == 'plateau':
            scheduler.step(avg_test_loss)
        else:
            scheduler.step()

        if args.teacher_forcing_decay > 0:
            tf_ratio = max(0.0, tf_ratio * args.teacher_forcing_decay)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % args.log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | LR: {current_lr:.6f}")

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"  Ранняя остановка на эпохе {epoch}")
            break

    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(run_dir, 'best_model.pth'))
    print(f"  Обучение завершено. Лучшая модель: эпоха {best_epoch}, Test Loss: {best_test_loss:.6f}")
        # Сохраняем результат в ExperimentManager
    if args.experiment_id:
        from experiment_manager import manager
        manager.update_status(
            args.experiment_id,
            "completed",
            best_test_loss=best_test_loss,
            best_epoch=best_epoch
        )

    # Сохраняем график и лог
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best model (epoch {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (normalized)')
    plt.title(f'Training curves: {model_name} ({run_timestamp})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'loss_plot.png'))
    plt.close()

    with open(os.path.join(run_dir, 'training_log.txt'), 'w') as f:
        f.write(f"JSON file: {json_path}\nRun timestamp: {run_timestamp}\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"Best epoch: {best_epoch} | Test Loss: {best_test_loss:.6f}\n")

    if args.result_file:
        result = {
            'experiment_id': args.experiment_id,
            'best_test_loss': best_test_loss,
            'best_epoch': best_epoch,
            'model_path': os.path.join(run_dir, 'best_model.pth'),
            'run_dir': run_dir,
            'status': 'completed'
        }
        with open(args.result_file, 'w') as f:
            json.dump(result, f)

    return best_test_loss, run_dir

def main():
    parser = argparse.ArgumentParser(description='Train RAE on a single JSON file')
    parser.add_argument('--json_file', type=str, help='Путь к конкретному JSON файлу')
    parser.add_argument('--output_base', type=str, default='models')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=40)
    parser.add_argument('--min_conf', type=float, default=None)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'])
    parser.add_argument('--autoregressive', action='store_true', default=True)
    parser.add_argument('--no_autoregressive', dest='autoregressive', action='store_false')
    parser.add_argument('--teacher_forcing', type=float, default=0.5)
    parser.add_argument('--teacher_forcing_decay', type=float, default=0.99)
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--result_file', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use')
    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)

    if args.json_file:
        train_model_for_json(args.json_file, args.output_base, args)
    else:
        print("Error: --json_file is required")
        sys.exit(1)

if __name__ == '__main__':
    main()