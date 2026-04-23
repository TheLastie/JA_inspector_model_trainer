# Документация системы обучения и анализа моделей движения

Версия 2.0 | Апрель 2026

## Оглавление

1. [Общее описание](#1-общее-описание)
2. [Структура проекта](#2-структура-проекта)
3. [Установка и зависимости](#3-установка-и-зависимости)
4. [Модули системы](#4-модули-системы)
   - 4.1. model_zoo.py – Архитектуры моделей
   - 4.2. data_pipeline.py – Подготовка данных
   - 4.3. experiment_manager.py – Управление экспериментами
   - 4.4. run_experiment.py – Запуск одного обучения
   - 4.5. massive_trainer.py – Массовое обучение
   - 4.6. summary_report.py – Генерация отчётов
   - 4.7. test_system.py – Тесты
5. [Команды запуска](#5-команды-запуска)
6. [Примеры использования](#6-примеры-использования)
7. [Тестирование](#7-тестирование)
8. [Часто задаваемые вопросы](#8-часто-задаваемые-вопросы)

---

## 1. Общее описание

Система предназначена для автоматического обучения и сравнения моделей машинного обучения на данных о движениях человека (упражнения). Она включает:

- Загрузку и предобработку данных из JSON (скелетные точки, векторы, углы)
- Четыре типа нейросетевых моделей (RAE, Transformer, VAE, Analyzer)
- Управление экспериментами через файловое хранилище
- Массовое обучение с перебором гиперпараметров
- Тестирование (pytest) и генерацию сводных отчётов

---

## 2. Структура проекта

```
project/
├── model_zoo.py              # Архитектуры моделей
├── data_pipeline.py          # Загрузка и подготовка данных
├── experiment_manager.py     # Управление метаданными экспериментов
├── run_experiment.py         # Запуск одного обучения
├── massive_trainer.py        # Оркестратор массового обучения
├── hyper_optim.py            # Подбор гиперпараметров (опционально)
├── summary_report.py         # Генерация отчётов
├── top_models_report.py      # Вывод лучших моделей
├── test_system.py            # pytest-тесты
├── datasets/                 # JSON-файлы с данными упражнений
├── models/                   # Сохранённые модели и артефакты
└── experiments/              # Метаданные экспериментов (создаётся автоматически)
```

---

## 3. Установка и зависимости

### Требования

- Python 3.11+
- PyTorch 2.0+
- CUDA (опционально, для GPU)

### Установка

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

# Установка зависимостей
pip install torch numpy matplotlib pytest
```

---

## 4. Модули системы

### 4.1. `model_zoo.py` – Архитектуры моделей

Содержит классы:

| Класс | Описание | Вход | Выход |
|-------|----------|------|-------|
| `RecurrentAutoencoder` | LSTM‑автоэнкодер с авторегрессионным декодером | (batch, seq, features) | (recon, latent) |
| `TransformerAutoencoder` | Автоэнкодер на основе Transformer | (batch, seq, features) | (recon, latent) |
| `LSTMVAE` | Вариационный автоэнкодер с LSTM | (batch, seq, features) | (recon, mu, logvar) |
| `ExerciseAnalyzer` | Бинарный классификатор ошибок по суставам | (batch, seq, features) | список тензоров (batch, 2) |

**Пример создания модели:**
```python
from model_zoo import RecurrentAutoencoder
model = RecurrentAutoencoder(input_dim=11, hidden_dim=128, latent_dim=64)
```

---

### 4.2. `data_pipeline.py` – Подготовка данных

**Функции:**

| Функция | Описание |
|---------|----------|
| `load_all_frames(json_path, use_full_features)` | Загружает все кадры из JSON, возвращает `np.ndarray` и границы видео |
| `split_by_videos(angles, boundaries, test_ratio, seq_len)` | Разделение на train/test с сохранением целостности видео |
| `SequenceDataset` | Создаёт окна фиксированной длины, выполняет z‑score нормализацию |
| `get_dataloaders(...)` | Возвращает `DataLoader` для обучения и теста |

**Режимы признаков:**

- `use_full_features=False` (по умолчанию) – только 11 углов.
- `use_full_features=True` – 44 признака: нормализованные векторы (22), нормализованные длины (11), углы (11).

**Пример:**
```python
from data_pipeline import get_dataloaders
train_loader, test_loader = get_dataloaders('datasets/squat.json', seq_len=30, batch_size=32)
```

---

### 4.3. `experiment_manager.py` – Управление экспериментами

Класс `ExperimentManager` хранит метаданные в папке `experiments/`. Каждый эксперимент – подпапка с `meta.json` и артефактами.

**Основные методы:**

| Метод | Описание |
|-------|----------|
| `create_experiment(model_type, exercise, params)` | Создаёт новый эксперимент, возвращает `exp_id` |
| `update_status(exp_id, status, **metrics)` | Обновляет статус и метрики |
| `save_artifact(exp_id, name, content)` | Сохраняет файл в папку эксперимента |
| `list_experiments(status, model_type, exercise)` | Возвращает список экспериментов с фильтрацией |
| `delete_experiment(exp_id)` | Удаляет папку эксперимента |

**Структура `meta.json`:**
```json
{
  "id": "a1b2c3d4",
  "model_type": "rae",
  "exercise": "squat",
  "params": {"lr": 0.001, "hidden_dim": 128},
  "status": "completed",
  "created_at": "2026-04-22T10:00:00",
  "started_at": "2026-04-22T10:00:01",
  "finished_at": "2026-04-22T10:15:30",
  "result": {"best_test_loss": 0.0123, "best_epoch": 42},
  "artifacts": {"best_model.pth": {...}}
}
```

---

### 4.4. `run_experiment.py` – Запуск одного обучения

Единый скрипт для обучения любой модели. Заменяет `train_rae_per_json.py` и `train_exercise_analyzer.py`.

**Аргументы командной строки:**

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--model_type` | str | **обязательно** | `rae`, `transformer`, `vae`, `analyzer` |
| `--json_file` | str | **обязательно** | Путь к JSON с данными |
| `--output_dir` | str | `models` | Папка для сохранения модели |
| `--experiment_id` | str | `None` | ID существующего эксперимента |
| `--seq_len` | int | 30 | Длина окна (кадров) |
| `--batch_size` | int | 32 | Размер батча |
| `--test_ratio` | float | 0.1 | Доля тестовой выборки |
| `--epochs` | int | 100 | Количество эпох |
| `--lr` | float | 0.005 | Learning rate |
| `--early_stop_patience` | int | 20 | Терпение для ранней остановки |
| `--hidden_dim` | int | 128 | Размер скрытого слоя LSTM |
| `--latent_dim` | int | 64 | Размер латентного вектора |
| `--num_layers` | int | 2 | Количество слоёв LSTM |
| `--dropout` | float | 0.2 | Dropout |
| `--autoregressive` | bool | True | Авторегрессионный декодер (для RAE) |
| `--d_model` | int | 128 | Размерность Transformer |
| `--nhead` | int | 4 | Число голов внимания |
| `--gpu_id` | int | 0 | ID GPU |
| `--use_full_features` | flag | False | Использовать 44 признака вместо 11 |

**Пример:**
```bash
python run_experiment.py --model_type rae --json_file datasets/lateral_raise.json --epochs 50 --gpu_id 0
```

---

### 4.5. `massive_trainer.py` – Массовое обучение

Оркестратор для запуска обучения на всех упражнениях с перебором гиперпараметров.

**Аргументы:**

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--model_type` | str | `rae` | `rae`, `transformer`, `vae`, `all` |
| `--workers` | int | 2 | Число параллельных процессов |
| `--epochs` | int | 100 | Количество эпох |
| `--seq_len` | int | 30 | Длина окна |
| `--batch_size` | int | 32 | Размер батча |
| `--lr` | float | 0.005 | Learning rate |
| `--hidden_dim` | int | 128 | Размер скрытого слоя |
| `--latent_dim` | int | 64 | Размер латентного вектора |
| `--use_full_features` | flag | False | Расширенные признаки |

**Пример:**
```bash
python massive_trainer.py --model_type all --workers 6 --epochs 50 --use_full_features
```

---

### 4.6. `summary_report.py` – Генерация отчётов

Анализирует завершённые эксперименты и строит сводные таблицы.

**Аргументы:**

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--top_n` | int | 3 | Количество лучших моделей для вывода |
| `--export_csv` | str | `None` | Сохранить отчёт в CSV |
| `--compare_architectures` | flag | False | Сравнение архитектур |
| `--status` | str | `completed` | Фильтр по статусу |
| `--exercise` | str | `None` | Фильтр по упражнению |

**Пример:**
```bash
python summary_report.py --top_n 5 --export_csv top_models.csv
```

---

### 4.7. `test_system.py` – Тесты

Набор pytest-тестов, покрывающих:

- `ExperimentManager` (создание, обновление, удаление)
- `data_pipeline` (загрузка JSON, разделение, датасет)
- Все модели (forward, backward, размерности)
- Граничные случаи (пустой JSON, короткие последовательности)

**Запуск:**
```bash
pytest test_system.py -v
```

---

## 5. Команды запуска

### 5.1. Запуск одного обучения

```bash
python run_experiment.py \
    --model_type rae \
    --json_file datasets/lateral_raise.json \
    --output_dir models \
    --seq_len 30 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.005 \
    --gpu_id 0
```

### 5.2. Массовое обучение

```bash
python massive_trainer.py \
    --model_type rae \
    --workers 6 \
    --epochs 50 \
    --seq_len 30 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --use_full_features
```

### 5.3. Генерация отчёта

```bash
python summary_report.py --top_n 3 --export_csv best_models.csv
```

### 5.4. Очистка экспериментов

```bash
# Удалить все failed эксперименты
python -c "from experiment_manager import manager; [manager.delete_experiment(e['id']) for e in manager.list_experiments(status='failed')]"
```

### 5.5. Запуск тестов

```bash
pytest test_system.py -v
```

---

## 6. Примеры использования

### Сценарий 1: Быстрое тестирование новой модели

```bash
# 1. Запустить RAE на 2 эпохах
python run_experiment.py --model_type rae --json_file datasets/lateral_raise.json --epochs 2

# 2. Проверить метрики
cat experiments/*/meta.json | grep best_test_loss
```

### Сценарий 2: Массовый поиск лучшей архитектуры

```bash
# Запустить все типы моделей на всех упражнениях с полными признаками
python massive_trainer.py --model_type all --workers 6 --use_full_features

# После завершения сгенерировать отчёт
python summary_report.py --top_n 5 --export_csv final_results.csv
```

### Сценарий 3: Сравнение моделей на одном упражнении

```bash
# Обучить три модели на одном JSON
for mt in rae transformer vae; do
    python run_experiment.py --model_type $mt --json_file datasets/squat.json --epochs 30
done

# Сравнить результаты
python summary_report.py --exercise squat --compare_architectures
```

---

## 7. Тестирование

Все модули покрыты тестами. Для запуска:

```bash
pytest test_system.py -v
```

Ожидаемый результат: **17 passed**.

При возникновении ошибок импорта убедитесь, что все файлы лежат в одной директории и активировано виртуальное окружение.

---

## 8. Часто задаваемые вопросы

### В: Почему используется только углы, а не векторы?
**О:** По умолчанию `use_full_features=False` – углы инвариантны к росту и расстоянию до камеры. Для задач, требующих пространственной информации (например, ширина хвата), включите `--use_full_features`.

### В: Как добавить новую модель?
**О:** 
1. Добавьте класс в `model_zoo.py`.
2. Зарегистрируйте в `get_model()` внутри `run_experiment.py`.
3. При необходимости обновите `massive_trainer.py`.

### В: Где хранятся результаты?
**О:** Метаданные в `experiments/*/meta.json`, модели в `models/` (или указанной через `--output_dir`).

### В: Как прервать массовое обучение?
**О:** Нажмите `Ctrl+C` – менеджер корректно завершит текущие задачи.

---
