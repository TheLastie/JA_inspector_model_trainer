# Спецификация моделей проекта

## Общая информация
- **Входные данные:** Последовательности углов (11 суставов) или расширенные признаки (44).
- **Нормализация:** Углы из радиан `[0, π]` преобразуются в `[-1, 1]` по формуле `(angles / π)*2 - 1`.
- **Loss для генеративных моделей:** MSE (для VAE добавляется KL-дивергенция с весом 0.001).
- **Метрика качества:** `best_test_loss` – MSE на тестовых видео.

## RecurrentAutoencoder

**Описание:** LSTM-автоэнкодер с авторегрессионным декодером. Энкодер сжимает последовательность в латентный вектор, декодер восстанавливает последовательность, используя предыдущие предсказания.

**Сигнатура конструктора:** `RecurrentAutoencoder(input_dim, hidden_dim=128, latent_dim=64, num_layers=2, dropout=0.2, autoregressive=True)`

**Вход:** Tensor (batch, seq_len, input_dim) – нормализованные углы (11) или 44 признака.

**Выход:** (reconstructed: Tensor, latent: Tensor). recon имеет ту же форму, что и вход.

**Типичные гиперпараметры:**
- `input_dim`: 11
- `hidden_dim`: 128
- `latent_dim`: 64
- `num_layers`: 2
- `dropout`: 0.2
- `autoregressive`: True

**Число параметров:** не удалось вычислить автоматически (RecurrentAutoencoder.__init__() missing 1 required positional argument: 'input_dim')

## TransformerAutoencoder

**Описание:** Трансформерный автоэнкодер. Использует позиционное кодирование, энкодер Transformer и декодер Transformer. Latent получается через среднее пулинг.

**Сигнатура конструктора:** `TransformerAutoencoder(input_dim, d_model=128, nhead=4, num_layers=3, latent_dim=64, seq_len=30, dropout=0.1)`

**Вход:** Tensor (batch, seq_len, input_dim)

**Выход:** (reconstructed: Tensor, latent: Tensor)

**Типичные гиперпараметры:**
- `input_dim`: 11
- `d_model`: 128
- `nhead`: 4
- `num_layers`: 3
- `latent_dim`: 64
- `seq_len`: 30
- `dropout`: 0.1

**Число параметров:** не удалось вычислить автоматически (TransformerAutoencoder.__init__() missing 1 required positional argument: 'input_dim')

## LSTMVAE

**Описание:** Вариационный LSTM-автоэнкодер. Выдаёт mu и logvar для репараметризации, добавляет KL-дивергенцию к MSE.

**Сигнатура конструктора:** `LSTMVAE(input_dim, hidden_dim=128, latent_dim=64, num_layers=2, dropout=0.2)`

**Вход:** Tensor (batch, seq_len, input_dim)

**Выход:** (reconstructed: Tensor, mu: Tensor, logvar: Tensor)

**Типичные гиперпараметры:**
- `input_dim`: 11
- `hidden_dim`: 128
- `latent_dim`: 64
- `num_layers`: 2
- `dropout`: 0.2

**Число параметров:** не удалось вычислить автоматически (LSTMVAE.__init__() missing 1 required positional argument: 'input_dim')

## ExerciseAnalyzer

**Описание:** Классификатор ошибок по 11 суставам. Требует предобученный энкодер. Возвращает список из 11 тензоров логитов формы (batch, 2).

**Сигнатура конструктора:** `ExerciseAnalyzer(encoder, latent_dim, num_joints=11, hidden_dim=128, dropout=0.3)`

**Вход:** Tensor (batch, seq_len, input_dim)

**Выход:** List[Tensor] – каждый тензор (batch, 2) для бинарной классификации сустава.

**Типичные гиперпараметры:**
- `encoder`: предобученный RAE/Transformer/VAE
- `latent_dim`: 64
- `num_joints`: 11
- `hidden_dim`: 128
- `dropout`: 0.3

**Число обучаемых параметров (типичная конфигурация):** 100,566


## Примечания
- Все модели ожидают нормализованные входные данные (z-score применяется в `data_pipeline.SequenceDataset`).
- `seq_len` – длина окна (обычно 30 кадров).
- Для обучения используйте `run_experiment.py` или `massive_trainer.py`.
- Результаты экспериментов хранятся в папке `experiments/` и управляются `ExperimentManager`.
