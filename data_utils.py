# data_utils.py
import json
import numpy as np
import torch
from torch.utils.data import Dataset

# Класс SplittedData (копия из вашего кода)
class SplittedData:
    def __init__(self, points):
        self.points = {i: (np.array(points[i][0]), points[i][1]) for i in points}
        self.body_vectors = [(0, 0, 0)] * 11
        self.angles = [0] * 11
        self.maked = 0
        self.make_vectors()
        self.make_angles()

    def make_vectors(self):
        self.maked = 1
        self.body_vectors[0] = self.points[2][0] - self.points[5][0]
        self.body_vectors[1] = self.points[2][0] - self.points[3][0]
        self.body_vectors[2] = self.points[3][0] - self.points[4][0]
        self.body_vectors[3] = self.points[5][0] - self.points[6][0]
        self.body_vectors[4] = self.points[6][0] - self.points[7][0]
        self.body_vectors[5] = self.points[1][0] - self.points[11][0]
        self.body_vectors[6] = self.points[1][0] - self.points[8][0]
        self.body_vectors[7] = self.points[8][0] - self.points[9][0]
        self.body_vectors[8] = self.points[11][0] - self.points[12][0]
        self.body_vectors[9] = self.points[12][0] - self.points[13][0]
        self.body_vectors[10] = self.points[9][0] - self.points[10][0]
        for i, val in enumerate(self.body_vectors):
            self.body_vectors[i] = val, np.linalg.norm(val)

    def angle_2_vec(self, vec1, vec2, vec1l=None, vec2l=None):
        if vec1l and vec2l:
            cos_val = np.clip(np.dot(vec1 / vec1l, vec2 / vec2l), -1.0, 1.0)
            return np.arccos(cos_val)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_val = np.clip(np.dot(vec1 / norm1, vec2 / norm2), -1.0, 1.0)
        return np.arccos(cos_val)

    def make_angles(self):
        if not self.maked:
            self.make_vectors()
        self.angles[0] = self.angle_2_vec(self.body_vectors[0][0], self.body_vectors[3][0],
                                          self.body_vectors[0][1], self.body_vectors[3][1])
        self.angles[1] = self.angle_2_vec(self.body_vectors[0][0], self.body_vectors[1][0],
                                          self.body_vectors[0][1], self.body_vectors[1][1])
        self.angles[2] = self.angle_2_vec(self.body_vectors[5][0], self.body_vectors[6][0],
                                          self.body_vectors[5][1], self.body_vectors[6][1])
        self.angles[3] = self.angle_2_vec(self.body_vectors[0][0], self.body_vectors[5][0],
                                          self.body_vectors[0][1], self.body_vectors[5][1])
        self.angles[4] = self.angle_2_vec(self.body_vectors[0][0], self.body_vectors[6][0],
                                          self.body_vectors[0][1], self.body_vectors[6][1])
        self.angles[5] = self.angle_2_vec(self.body_vectors[1][0], self.body_vectors[2][0],
                                          self.body_vectors[1][1], self.body_vectors[2][1])
        self.angles[6] = self.angle_2_vec(self.body_vectors[6][0], self.body_vectors[7][0],
                                          self.body_vectors[6][1], self.body_vectors[7][1])
        self.angles[7] = self.angle_2_vec(self.body_vectors[5][0], self.body_vectors[8][0],
                                          self.body_vectors[5][1], self.body_vectors[8][1])
        self.angles[8] = self.angle_2_vec(self.body_vectors[8][0], self.body_vectors[9][0],
                                          self.body_vectors[8][1], self.body_vectors[9][1])
        self.angles[9] = self.angle_2_vec(self.body_vectors[7][0], self.body_vectors[10][0],
                                          self.body_vectors[7][1], self.body_vectors[10][1])
        self.angles[10] = self.angle_2_vec(self.body_vectors[3][0], self.body_vectors[4][0],
                                           self.body_vectors[3][1], self.body_vectors[4][1])

    def get_features(self):
        """Возвращает расширенный вектор признаков для кадра."""
        coords = np.array([self.points[i][0] for i in range(1, 14)])  # 13 точек
        vecs = np.array([v[0] for v in self.body_vectors])            # 11 векторов
        lengths = np.array([v[1] for v in self.body_vectors])         # 11 длин
        angles = np.array(self.angles)                                # 11 углов
        confs = np.array([self.points[i][1] for i in range(1, 14)])   # уверенности
        return np.concatenate([coords.flatten(), vecs.flatten(), lengths, angles, confs])

def load_angles_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    all_features = []
    video_boundaries = []
    current_idx = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "vids_list": continue
            if isinstance(value, dict):
                sorted_frames = sorted(value.items(), key=lambda x: int(x[0]))
                video_features = []
                for _, fd in sorted_frames:
                    if 'vectors' in fd and 'angles' in fd:
                        # Реконструируем points из vectors (обратное преобразование сложное, поэтому используем готовые признаки)
                        # Для простоты используем только angles (можно расширить)
                        video_features.append(fd['angles'])
                if video_features:
                    video_boundaries.append((current_idx, current_idx + len(video_features)))
                    all_features.extend(video_features)
                    current_idx += len(video_features)
    elif isinstance(data, list):
        video_features = [item['angles'] for item in data if 'angles' in item]
        if video_features:
            video_boundaries.append((0, len(video_features)))
            all_features = video_features
    return np.array(all_features, dtype=np.float32), video_boundaries

class SequenceDataset(Dataset):
    def __init__(self, data, seq_len=30, normalize=True):
        self.data = data.astype(np.float32)
        if normalize:
            self.data = (self.data - self.data.mean(axis=0)) / (self.data.std(axis=0) + 1e-8)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx+self.seq_len])