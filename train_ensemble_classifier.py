#!/usr/bin/env python3
"""
train_ensemble_classifier.py – Обучение классификатора упражнений на основе ансамбля автоэнкодеров.
Динамически определяет все упражнения с готовыми моделями.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

from model_zoo import RecurrentAutoencoder, TransformerAutoencoder, LSTMVAE
from experiment_manager import manager
from data_pipeline import load_all_frames


def find_best_model_for_exercise(exercise: str, models_base_dir: str = 'models'):
    experiments = manager.list_experiments(status='completed')
    candidates = []
    for exp in experiments:
        if exp['exercise'] != exercise:
            continue
        if 'best_test_loss' not in exp.get('result', {}):
            continue
        candidates.append(exp)
    if not candidates:
        return None, None
    best_exp = min(candidates, key=lambda e: e['result']['best_test_loss'])
    exp_dir = manager.get_experiment_dir(best_exp['id'])
    model_path = os.path.join(models_base_dir, exp_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        return None, None
    return model_path, best_exp


def load_model(model_path: str, model_type: str, device: torch.device, metadata: dict) -> nn.Module:
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    params = metadata.get('params', {})

    if model_type == 'rae':
        model = RecurrentAutoencoder(
            input_dim=params.get('input_dim', 11),
            hidden_dim=params.get('hidden_dim', 128),
            latent_dim=params.get('latent_dim', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            autoregressive=params.get('autoregressive', True)
        )
    elif model_type == 'vae':
        model = LSTMVAE(
            input_dim=params.get('input_dim', 11),
            hidden_dim=params.get('hidden_dim', 128),
            latent_dim=params.get('latent_dim', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2)
        )
    elif model_type == 'transformer':
        model = TransformerAutoencoder(
            input_dim=params.get('input_dim', 11),
            d_model=params.get('d_model', 128),
            nhead=params.get('nhead', 4),
            num_layers=params.get('num_layers', 3),
            latent_dim=params.get('latent_dim', 64),
            seq_len=params.get('seq_len', 30),
            dropout=params.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class WindowDataset(Dataset):
    def __init__(self, data_list, seq_len=30, normalize=True, stride_ratio=0.5):
        self.samples = []
        for angles in data_list:
            if len(angles) < seq_len:
                continue
            if normalize:
                angles = (angles / np.pi) * 2.0 - 1.0
            stride = max(1, int(seq_len * stride_ratio))
            for start in range(0, len(angles) - seq_len + 1, stride):
                self.samples.append(angles[start:start+seq_len])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx])


def compute_ensemble_features(ensemble_models, dataset, batch_size=256, device=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    features = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing features"):
            batch = batch.to(device)
            batch_feats = []
            for name, model in ensemble_models.items():
                if isinstance(model, LSTMVAE):
                    recon, _, _ = model(batch)
                else:
                    recon, _ = model(batch)

                mse = torch.mean((recon - batch) ** 2, dim=(1, 2)).cpu().numpy()
                mae = torch.mean(torch.abs(recon - batch), dim=(1, 2)).cpu().numpy()
                max_err = torch.max(torch.abs(recon - batch), dim=2)[0].mean(dim=1).cpu().numpy()
                var_mse = torch.var(torch.mean((recon - batch) ** 2, dim=2), dim=1).cpu().numpy()

                batch_feats.extend([mse, mae, max_err, var_mse])
            # Отношения ошибок между моделями
            num_models = len(ensemble_models)
            for i in range(num_models):
                for j in range(i+1, num_models):
                    ratio = batch_feats[i*4] / (batch_feats[j*4] + 1e-8)
                    batch_feats.append(ratio)
            features.append(np.stack(batch_feats, axis=1))

    return np.vstack(features)


def load_all_labeled_data(data_dir='datasets', known_exercises=None):
    all_angles = []
    all_labels = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(data_dir, fname)
        ex_name = os.path.splitext(fname)[0]
        label = ex_name if ex_name in known_exercises else 'other'
        try:
            angles, _ = load_all_frames(path, min_confidence=None)
            if len(angles) > 0:
                all_angles.append(angles)
                all_labels.append(label)
                print(f"Loaded {fname}: {len(angles)} frames -> label '{label}'")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return all_angles, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='datasets')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', default='models/ensemble')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.10)
    parser.add_argument('--xgb_rounds', type=int, default=300)
    parser.add_argument('--xgb_lr', type=float, default=0.03)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Находим все упражнения с обученными моделями
    experiments = manager.list_experiments(status='completed')
    available_exercises = set()
    for exp in experiments:
        if 'best_test_loss' in exp.get('result', {}):
            available_exercises.add(exp['exercise'])
    available_exercises = sorted(list(available_exercises))
    print(f"Detected exercises with models: {available_exercises}")

    # 2. Загружаем модели для всех доступных упражнений
    ensemble_models = {}
    model_metadata = {}
    for ex in available_exercises:
        path, meta = find_best_model_for_exercise(ex)
        if path is None:
            continue
        print(f"Loading {ex} model from {path}")
        model = load_model(path, meta['model_type'], device, meta)
        ensemble_models[ex] = model
        model_metadata[ex] = meta

    if not ensemble_models:
        raise RuntimeError("No models found for ensemble.")

    # 3. Загружаем данные и создаём датасет окон
    all_angles, all_labels = load_all_labeled_data(args.data_dir, set(ensemble_models.keys()))
    window_dataset = WindowDataset(all_angles, seq_len=args.seq_len)
    print(f"Total windows: {len(window_dataset)}")

    # 4. Вычисляем признаки (ошибки)
    features = compute_ensemble_features(ensemble_models, window_dataset, args.batch_size, device)
    print(f"Features shape: {features.shape}")

    # 5. Формируем метки для каждого окна
    all_window_labels = []
    for angles, label in zip(all_angles, all_labels):
        if len(angles) < args.seq_len:
            continue
        stride = max(1, args.seq_len // 2)
        n_windows = (len(angles) - args.seq_len) // stride + 1
        all_window_labels.extend([label] * n_windows)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(all_window_labels)
    print("Label mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

    # 6. Разделение на train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, y, test_size=args.val_ratio + args.test_ratio, random_state=42, stratify=y
    )
    val_size = args.val_ratio / (args.val_ratio + args.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_size, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 7. Обучение XGBoost
    class_counts = np.bincount(y_train)
    scale_pos_weights = [class_counts.max() / (c + 1) for c in class_counts]
    sample_weights = np.array([scale_pos_weights[label] for label in y_train])

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'max_depth': 6,
        'eta': args.xgb_lr,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_state': 42,
    }

    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    print("Training XGBoost...")
    bst = xgb.train(
        params, dtrain, num_boost_round=args.xgb_rounds,
        evals=evals, evals_result=evals_result,
        early_stopping_rounds=20, verbose_eval=10
    )

    probs = bst.predict(dtest)  # вернёт вероятности
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 8. Сохранение
    os.makedirs(args.output_dir, exist_ok=True)
    bst.save_model(os.path.join(args.output_dir, 'ensemble_classifier_xgb.json'))
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(args.output_dir, 'label_encoder.joblib'))
    with open(os.path.join(args.output_dir, 'model_metadata.json'), 'w') as f:
        json.dump({
            'model_names': list(ensemble_models.keys()),
            'model_metadata': model_metadata,
        }, f, indent=2)

    # 9. Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(evals_result['train']['mlogloss'], label='Train')
    plt.plot(evals_result['val']['mlogloss'], label='Validation')
    plt.xlabel('Boosting Round')
    plt.ylabel('Multi-class Log Loss')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'learning_curve.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    importance = bst.get_score(importance_type='gain')
    imp_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'gain': list(importance.values())
    }).sort_values('gain', ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['feature'], imp_df['gain'])
    plt.xlabel('Gain')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'feature_importance.png'), dpi=150)
    plt.close()

    print("All done. Model and visualizations saved.")


if __name__ == '__main__':
    main()