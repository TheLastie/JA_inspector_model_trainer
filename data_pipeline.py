import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

def load_raw_frames(json_path: str) -> Tuple[List[dict], List[Tuple[int, int]]]:
    """Загружает сырые кадры с полями 'vectors' и 'angles'."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_frames = []
    boundaries = []
    current_idx = 0

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "vids_list":
                continue
            if isinstance(value, dict):
                sorted_frames = sorted(value.items(), key=lambda x: int(x[0]))
                video_frames = [fd for _, fd in sorted_frames if 'vectors' in fd and 'angles' in fd]
            elif isinstance(value, list):
                video_frames = [item for item in value if 'vectors' in item and 'angles' in item]
            else:
                continue

            if video_frames:
                boundaries.append((current_idx, current_idx + len(video_frames)))
                all_frames.extend(video_frames)
                current_idx += len(video_frames)

    elif isinstance(data, list):
        video_frames = [item for item in data if 'vectors' in item and 'angles' in item]
        if video_frames:
            boundaries.append((0, len(video_frames)))
            all_frames = video_frames
    else:
        raise ValueError(f"Неподдерживаемый формат JSON: {type(data)}")

    return all_frames, boundaries


def extract_features(frame: dict, use_full_features: bool = False) -> np.ndarray:
    """
    Извлекает признаки из кадра.
    Если use_full_features=False: только углы (11).
    Если True: нормализованные векторы (11*2=22), длины сегментов (11), углы (11). Всего 44.
    """
    angles = np.array(frame['angles'], dtype=np.float32)
    if not use_full_features:
        return angles

    vectors_data = frame['vectors']  # список: [ [[x,y], length], ... ]
    num_vectors = len(vectors_data)
    vecs = np.zeros((num_vectors, 2), dtype=np.float32)
    lengths = np.zeros(num_vectors, dtype=np.float32)
    for i, v in enumerate(vectors_data):
        vecs[i] = v[0]
        lengths[i] = v[1]

    # Нормализация векторов до единичной длины
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    norm_vecs = vecs / norms

    # Нормализация длин относительно среднего по кадру
    mean_len = np.mean(lengths) if np.mean(lengths) > 0 else 1.0
    norm_lengths = lengths / mean_len

    features = np.concatenate([norm_vecs.flatten(), norm_lengths, angles])
    return features.astype(np.float32)


def load_all_frames(json_path: str, min_confidence: Optional[float] = None,
                    use_full_features: bool = False) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    all_frames, boundaries = load_raw_frames(json_path)
    if not all_frames:
        return np.array([], dtype=np.float32), []
    features = [extract_features(f, use_full_features) for f in all_frames]
    return np.array(features, dtype=np.float32), boundaries


def split_by_videos(angles: np.ndarray, boundaries: List[Tuple[int, int]],
                    test_ratio: float = 0.1, seq_len: int = 30) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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

    if len(train_data) < seq_len:
        train_data = None
    if len(test_data) < seq_len:
        test_data = None

    return train_data, test_data


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int = 30, normalize: bool = True):
        self.seq_len = seq_len
        self.data = data.copy().astype(np.float32)
        if normalize and len(self.data) > 0:
            # z-score нормализация по признакам
            mean = self.data.mean(axis=0, keepdims=True)
            std = self.data.std(axis=0, keepdims=True) + 1e-8
            self.data = (self.data - mean) / std

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx + self.seq_len])


def get_dataloaders(json_path: str, seq_len: int = 30, batch_size: int = 32,
                    test_ratio: float = 0.1, normalize: bool = True,
                    min_confidence: Optional[float] = None,
                    use_full_features: bool = False) -> Tuple[DataLoader, Optional[DataLoader]]:
    angles, boundaries = load_all_frames(json_path, min_confidence, use_full_features)
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