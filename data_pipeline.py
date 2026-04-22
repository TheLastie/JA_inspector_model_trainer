"""
Модуль data_pipeline.py – загрузка и предобработка данных для всех моделей.

Функции:
- load_all_frames: извлечение углов и границ видео из JSON.
- split_by_videos: разделение на train/test с сохранением целостности видео.
- SequenceDataset: создание окон заданной длины с нормализацией.
- get_dataloaders: фабрика для получения train/test DataLoader'ов.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Optional


def load_all_frames(json_path: str, min_confidence: Optional[float] = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Загружает все кадры из JSON-файла, извлекая поле 'angles'.
    Возвращает:
        angles_array: np.ndarray формы (total_frames, num_joints)
        boundaries: список кортежей (start_idx, end_idx) для каждого видео
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_angles = []
    boundaries = []
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
                boundaries.append((current_idx, current_idx + len(video_angles)))
                all_angles.extend(video_angles)
                current_idx += len(video_angles)

    elif isinstance(data, list):
        video_angles = [item['angles'] for item in data if 'angles' in item and frame_passes(item)]
        if video_angles:
            boundaries.append((0, len(video_angles)))
            all_angles = video_angles
    else:
        raise ValueError(f"Неподдерживаемый формат JSON: {type(data)}")

    if not all_angles:
        return np.array([], dtype=np.float32), []

    return np.array(all_angles, dtype=np.float32), boundaries


def split_by_videos(
    angles: np.ndarray,
    boundaries: List[Tuple[int, int]],
    test_ratio: float = 0.1,
    seq_len: int = 30
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Разделяет данные на train и test, гарантируя, что кадры одного видео не попадут в обе выборки.

    Args:
        angles: массив всех углов (total_frames, num_joints)
        boundaries: границы видео
        test_ratio: доля кадров для теста (приблизительно)
        seq_len: минимальная длина последовательности (используется для проверки достаточности данных)

    Returns:
        train_data: np.ndarray или None
        test_data: np.ndarray или None
    """
    np.random.seed(42)
    video_indices = list(range(len(boundaries)))
    np.random.shuffle(video_indices)

    total_frames = len(angles)
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
        frames = [angles[s:e] for s, e in video_list]
        if frames:
            return np.concatenate(frames, axis=0)
        return np.array([], dtype=np.float32)

    train_data = collect_frames(train_videos)
    test_data = collect_frames(test_videos)

    # Если после разделения данных недостаточно для создания хотя бы одного окна, возвращаем None
    if len(train_data) < seq_len:
        train_data = None
    if len(test_data) < seq_len:
        test_data = None

    return train_data, test_data


class SequenceDataset(Dataset):
    """Создаёт окна фиксированной длины из временного ряда."""
    def __init__(self, data: np.ndarray, seq_len: int = 30, normalize: bool = True):
        """
        Args:
            data: массив (num_frames, num_features)
            seq_len: длина окна
            normalize: если True, данные нормализуются к диапазону [-1, 1] (для углов [0, π])
        """
        self.seq_len = seq_len
        self.data = data.copy().astype(np.float32)
        if normalize and len(self.data) > 0:
            # Нормализация: предполагаем, что углы в радианах [0, π] -> [-1, 1]
            self.data = (self.data / np.pi) * 2.0 - 1.0

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx + self.seq_len])


def get_dataloaders(
    json_path: str,
    seq_len: int = 30,
    batch_size: int = 32,
    test_ratio: float = 0.1,
    normalize: bool = True,
    min_confidence: Optional[float] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Фабрика для создания train и test DataLoader'ов из JSON-файла.

    Returns:
        train_loader, test_loader
    """
    angles, boundaries = load_all_frames(json_path, min_confidence)
    if len(angles) == 0:
        raise ValueError("Нет данных после фильтрации")

    train_data, test_data = split_by_videos(angles, boundaries, test_ratio, seq_len)

    if train_data is None:
        raise ValueError("Недостаточно данных для обучающей выборки")

    train_dataset = SequenceDataset(train_data, seq_len, normalize)
    test_dataset = SequenceDataset(test_data, seq_len, normalize) if test_data is not None else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    return train_loader, test_loader