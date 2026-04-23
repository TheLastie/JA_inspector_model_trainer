#!/usr/bin/env python3
"""
Обучение классификатора типов упражнений с fine‑tuning предобученного энкодера.
Поддерживаемые упражнения: squat, shoulder_press, push-up, pull-up, other.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from model_zoo import RecurrentAutoencoder, TransformerAutoencoder, LSTMVAE
from experiment_manager import manager
from data_pipeline import load_all_frames

EXERCISE_CLASSES = ['squat', 'shoulder_press', 'push-up', 'pull-up', 'other']


class ExerciseClassifier(nn.Module):
    """Классификатор с возможностью разморозки энкодера."""
    def __init__(self, encoder: nn.Module, latent_dim: int, num_classes: int, hidden_dim: int = 128,
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

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
        if isinstance(self.encoder, LSTMVAE):
            mu, _ = self.encoder.encode(x)
            latent = mu
        else:
            _, latent = self.encoder(x)
        return self.classifier(latent)


class LabeledSequenceDataset(Dataset):
    def __init__(self, data_list, labels, seq_len=30, normalize=True, augment=False):
        self.seq_len = seq_len
        self.normalize = normalize
        self.augment = augment
        self.samples = []
        self.labels = []

        for angles, label in zip(data_list, labels):
            if len(angles) < seq_len:
                continue
            if normalize:
                angles = (angles / np.pi) * 2.0 - 1.0
            stride = max(1, seq_len // 2)
            for start in range(0, len(angles) - seq_len + 1, stride):
                self.samples.append(angles[start:start+seq_len])
                self.labels.append(label)

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()
        if self.augment and np.random.rand() > 0.5:
            x = x[::-1].copy()
        return torch.FloatTensor(x), torch.tensor(self.encoded_labels[idx], dtype=torch.long)


def load_all_labeled_data(data_dir='datasets', exercises=EXERCISE_CLASSES):
    all_angles = []
    all_labels = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(data_dir, fname)
        ex_name = os.path.splitext(fname)[0]
        label = ex_name if ex_name in exercises else 'other'
        try:
            angles, _ = load_all_frames(path, min_confidence=None)
            if len(angles) > 0:
                all_angles.append(angles)
                all_labels.append(label)
                print(f"Loaded {fname}: {len(angles)} frames -> label '{label}'")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return all_angles, all_labels


def train_classifier(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Сбор всех меток тренировочного набора для вычисления весов классов
    all_labels = []
    for _, y in train_loader.dataset:
        all_labels.append(y.item())
    unique_labels = np.unique(all_labels)

    class_weights_present = compute_class_weight('balanced', classes=unique_labels, y=all_labels)
    num_total_classes = model.classifier[-1].out_features
    full_weights = np.ones(num_total_classes)
    for i, cls_idx in enumerate(unique_labels):
        full_weights[cls_idx] = class_weights_present[i]

    class_weights = torch.FloatTensor(full_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    model.load_state_dict(best_state)
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_exp_id', required=True)
    parser.add_argument('--data_dir', default='datasets')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights (no fine-tuning)')
    parser.add_argument('--output_dir', default='models/classifier')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--augment', action='store_true', help='Apply time-reversal augmentation')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Загрузка энкодера
    exp = manager.load_experiment(args.encoder_exp_id)
    if exp is None:
        raise ValueError(f"Experiment {args.encoder_exp_id} not found")
    model_type = exp['model_type']
    params = exp['params']
    exp_dir = manager.get_experiment_dir(exp['id'])
    model_path = os.path.join('models', exp_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    input_dim = params.get('input_dim', 11)

    if model_type == 'rae':
        encoder = RecurrentAutoencoder(
            input_dim=input_dim,
            hidden_dim=params.get('hidden_dim', 128),
            latent_dim=params.get('latent_dim', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            autoregressive=params.get('autoregressive', True)
        )
    elif model_type == 'transformer':
        encoder = TransformerAutoencoder(
            input_dim=input_dim,
            d_model=params.get('d_model', 128),
            nhead=params.get('nhead', 4),
            num_layers=params.get('num_layers', 3),
            latent_dim=params.get('latent_dim', 64),
            seq_len=args.seq_len,
            dropout=params.get('dropout', 0.1)
        )
    elif model_type == 'vae':
        encoder = LSTMVAE(
            input_dim=input_dim,
            hidden_dim=params.get('hidden_dim', 128),
            latent_dim=params.get('latent_dim', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    state_dict = torch.load(model_path, map_location='cpu')
    encoder.load_state_dict(state_dict)
    print(f"Loaded encoder from {model_path}")

    # Загрузка данных
    all_angles, all_labels = load_all_labeled_data(args.data_dir, EXERCISE_CLASSES)
    if len(all_angles) == 0:
        raise RuntimeError("No data found")

    full_dataset = LabeledSequenceDataset(all_angles, all_labels, args.seq_len, augment=args.augment)
    print(f"Total windows: {len(full_dataset)}")
    print("Label mapping:", full_dataset.label_encoder.classes_)
    num_classes = len(full_dataset.label_encoder.classes_)

    # Создание модели с корректным числом классов
    latent_dim = params.get('latent_dim', 64)
    classifier = ExerciseClassifier(encoder, latent_dim, num_classes=num_classes,
                                    freeze_encoder=args.freeze_encoder)

    # Разделение на train/val
    indices = np.arange(len(full_dataset))
    labels_encoded = full_dataset.encoded_labels
    train_idx, val_idx = train_test_split(indices, test_size=args.val_ratio, random_state=42, stratify=labels_encoded)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    best_acc = train_classifier(classifier, train_loader, val_loader, device, args.epochs, args.lr)
    print(f"Best validation accuracy: {best_acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'exercise_classifier.pth')
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'label_encoder': full_dataset.label_encoder,
        'encoder_exp_id': args.encoder_exp_id,
        'params': vars(args),
    }, save_path)
    print(f"Classifier saved to {save_path}")


if __name__ == '__main__':
    main()