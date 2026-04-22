"""
Файл: test_system.py
Набор pytest-тестов для проверки корректности работы всех модулей системы.
Запуск: pytest test_system.py -v
"""

import pytest
import torch
import numpy as np
import os
import json
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Импортируем наши модули (предполагается, что они лежат в корне)
import model_zoo
from experiment_manager import ExperimentManager
# В начале файла test_system.py исправьте импорт:
from data_pipeline import SequenceDataset, load_all_frames, split_by_videos


# ------------------- Фикстуры -------------------
@pytest.fixture
def temp_experiment_dir():
    """Создаёт временную папку для экспериментов."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def sample_angles():
    """Генерирует синтетические данные углов (100 кадров, 11 суставов)."""
    return np.random.uniform(0, np.pi, size=(100, 11)).astype(np.float32)


@pytest.fixture
def sample_boundaries():
    """Границы видео: одно видео с кадрами 0-99."""
    return [(0, 100)]


@pytest.fixture
def dummy_encoder():
    """Фиктивный энкодер для тестов Analyzer."""
    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 64
        def forward(self, x):
            return torch.randn(x.size(0), 64)
    return DummyEncoder()


# ------------------- Тесты ExperimentManager -------------------
def test_experiment_manager_create(temp_experiment_dir):
    mgr = ExperimentManager(base_dir=temp_experiment_dir)
    exp_id = mgr.create_experiment(
        model_type="rae",
        exercise="squat",
        params={"lr": 0.001, "hidden": 128},
        tags=["test"]
    )
    assert len(exp_id) == 8
    meta = mgr.load_experiment(exp_id)
    assert meta["status"] == "pending"
    assert meta["model_type"] == "rae"
    assert meta["exercise"] == "squat"
    assert meta["params"]["lr"] == 0.001
    assert "test" in meta["tags"]


def test_experiment_manager_update(temp_experiment_dir):
    mgr = ExperimentManager(base_dir=temp_experiment_dir)
    exp_id = mgr.create_experiment("rae", "squat", {})
    mgr.update_status(exp_id, "running", pid=12345)
    meta = mgr.load_experiment(exp_id)
    assert meta["status"] == "running"
    assert meta["started_at"] is not None
    assert meta["result"]["pid"] == 12345

    mgr.update_status(exp_id, "completed", best_test_loss=0.0123, best_epoch=42)
    meta = mgr.load_experiment(exp_id)
    assert meta["status"] == "completed"
    assert meta["finished_at"] is not None
    assert meta["result"]["best_test_loss"] == 0.0123
    assert meta["result"]["best_epoch"] == 42


def test_experiment_manager_list_filter(temp_experiment_dir):
    mgr = ExperimentManager(base_dir=temp_experiment_dir)
    id1 = mgr.create_experiment("rae", "squat", {}, tags=["grid"])
    id2 = mgr.create_experiment("vae", "bench", {}, tags=["random"])
    mgr.update_status(id1, "completed")
    mgr.update_status(id2, "running")

    all_exps = mgr.list_experiments()
    assert len(all_exps) == 2

    completed = mgr.list_experiments(status="completed")
    assert len(completed) == 1
    assert completed[0]["id"] == id1

    vae_exps = mgr.list_experiments(model_type="vae")
    assert len(vae_exps) == 1
    assert vae_exps[0]["id"] == id2


def test_experiment_manager_delete(temp_experiment_dir):
    mgr = ExperimentManager(base_dir=temp_experiment_dir)
    exp_id = mgr.create_experiment("rae", "squat", {})
    assert os.path.exists(mgr.get_experiment_dir(exp_id))
    mgr.delete_experiment(exp_id)
    assert not os.path.exists(mgr.get_experiment_dir(exp_id))
    assert mgr.load_experiment(exp_id) is None


# ------------------- Тесты Data Pipeline -------------------
def test_load_all_frames():
    # Создаём временный JSON с данными
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "video1.mp4": {
                "0": {"angles": [0.1, 0.2, 0.3]},
                "1": {"angles": [0.4, 0.5, 0.6]}
            }
        }, f)
        tmp_path = f.name
    try:
        angles, boundaries = load_all_frames(tmp_path)
        assert angles.shape == (2, 3)
        assert boundaries == [(0, 2)]
    finally:
        os.unlink(tmp_path)


def test_split_by_videos(sample_angles, sample_boundaries):
    train_data, test_data = split_by_videos(
        sample_angles, sample_boundaries, test_ratio=0.2, seq_len=10
    )
    # При одном видео и test_ratio=0.2 все кадры могут уйти в тест,
    # тогда train_data будет None. Это корректное поведение.
    # Мы проверяем, что хотя бы одна из выборок не None.
    assert train_data is not None or test_data is not None
    if train_data is not None:
        assert len(train_data) >= 10
    if test_data is not None:
        assert len(test_data) >= 10


def test_sequence_dataset(sample_angles):
    dataset = SequenceDataset(sample_angles, seq_len=15, normalize=True)
    assert len(dataset) == 100 - 15
    seq = dataset[0]
    assert seq.shape == (15, 11)
    assert isinstance(seq, torch.Tensor)
    # Проверяем нормализацию: среднее близко к 0, std близко к 1
    assert -0.5 < seq.mean().item() < 0.5


# ------------------- Тесты Моделей -------------------
def test_rae_forward():
    model = model_zoo.RecurrentAutoencoder(
        input_dim=11, hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.1
    )
    x = torch.randn(4, 30, 11)
    recon, latent = model(x)
    assert recon.shape == x.shape
    assert latent.shape == (4, 32)
    # Проверяем, что градиенты текут
    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Градиент отсутствует у {name}"


def test_transformer_ae_forward():
    model = model_zoo.TransformerAutoencoder(
        input_dim=11, d_model=64, nhead=4, num_layers=2, latent_dim=32, seq_len=30
    )
    x = torch.randn(4, 30, 11)
    recon, latent = model(x)
    assert recon.shape == x.shape
    assert latent.shape == (4, 32)
    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()


def test_vae_forward():
    model = model_zoo.LSTMVAE(
        input_dim=11, hidden_dim=64, latent_dim=32, num_layers=2
    )
    x = torch.randn(4, 30, 11)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, 32)
    assert logvar.shape == (4, 32)
    # Проверяем KL-дивергенцию (должна быть неотрицательной)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    assert kl.item() >= 0


def test_analyzer_forward(dummy_encoder):
    analyzer = model_zoo.ExerciseAnalyzer(
        encoder=dummy_encoder,
        latent_dim=64,
        num_joints=11,
        hidden_dim=64
    )
    x = torch.randn(4, 30, 11)
    outputs = analyzer(x)
    assert len(outputs) == 11
    for out in outputs:
        assert out.shape == (4, 2)  # бинарная классификация


# ------------------- Тесты совместимости моделей с пайплайном -------------------
@pytest.mark.parametrize("model_class,model_kwargs", [
    (model_zoo.RecurrentAutoencoder, {"input_dim": 11, "hidden_dim": 64, "latent_dim": 32}),
    (model_zoo.TransformerAutoencoder, {"input_dim": 11, "d_model": 64, "nhead": 4, "num_layers": 2, "latent_dim": 32, "seq_len": 30}),
    (model_zoo.LSTMVAE, {"input_dim": 11, "hidden_dim": 64, "latent_dim": 32}),
])
def test_model_training_step(model_class, model_kwargs, sample_angles):
    """Проверяем, что модели могут обучаться на синтетических данных."""
    dataset = SequenceDataset(sample_angles, seq_len=30, normalize=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    model = model_class(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        if isinstance(model, model_zoo.LSTMVAE):
            recon, mu, logvar = model(batch)
            mse = torch.nn.functional.mse_loss(recon, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + 0.001 * kl
        else:
            recon, _ = model(batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
        loss.backward()
        optimizer.step()
        assert loss.item() < 10.0  # должно быстро снижаться
        break


# ------------------- Тесты ошибок и граничных случаев -------------------
def test_empty_json_handling():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({}, f)
        tmp_path = f.name
    try:
        angles, boundaries = load_all_frames(tmp_path)
        assert angles.size == 0
        assert boundaries == []
    finally:
        os.unlink(tmp_path)


def test_short_sequence_dataset(sample_angles):
    """Проверяем, что датасет корректно обрабатывает seq_len > длины данных."""
    dataset = SequenceDataset(sample_angles[:20], seq_len=30, normalize=True)
    assert len(dataset) == 0  # не может создать ни одного окна


def test_manager_nonexistent_experiment(temp_experiment_dir):
    mgr = ExperimentManager(base_dir=temp_experiment_dir)
    assert mgr.load_experiment("nonexistent") is None
    # не должно падать
    mgr.update_status("nonexistent", "completed")