# Спецификации моделей

## Общие положения
Все модели ожидают на вход нормализованные последовательности углов (форма `(batch, seq_len, num_angles)`).  
Для генеративных моделей (RAE, TransformerAE, VAE) выход – восстановленная последовательность той же формы. Loss – MSE.  
Analyzer ожидает предобученный энкодер и возвращает логиты для бинарной классификации каждого сустава.

### Нормализация
Входные углы (радианы, диапазон [0, π]) преобразуются в [-1, 1] по формуле:
`normalized = (angles / π) * 2 - 1`

### Размерности
- `num_angles = 11` (определяется классами SplittedData).
- `seq_len` – параметр (обычно 15–45 кадров).

---

## RAE (RecurrentAutoencoder)
**Вход:** `(batch, seq_len, num_angles)`  
**Выход:** `(reconstructed, latent)`  
**Latent:** `(batch, latent_dim)`  
**Параметры:** hidden_dim, latent_dim, num_layers, dropout, autoregressive

**Назначение:** сжатие и восстановление движений; latent-вектор используется для обнаружения аномалий.

---

## TransformerAE (TransformerAutoencoder)
**Вход:** `(batch, seq_len, num_angles)`  
**Выход:** `(reconstructed, latent)`  
**Latent:** `(batch, latent_dim)`  
**Параметры:** d_model, nhead, num_layers, latent_dim, dropout

**Особенности:** использует позиционное кодирование, global pooling по времени.

---

## LSTMVAE (Variational Autoencoder)
**Вход:** `(batch, seq_len, num_angles)`  
**Выход:** `(reconstructed, mu, logvar)`  
**Latent:** `(batch, latent_dim)` (сэмплируется из N(mu, std))  
**Параметры:** hidden_dim, latent_dim, num_layers, dropout

**Особенности:** KL-дивергенция добавляется к MSE, позволяет оценивать вероятность принадлежности к распределению нормальных движений.

---

## ExerciseAnalyzer
**Вход:** `(batch, seq_len, num_angles)`  
**Выход:** список из `num_joints` тензоров `(batch, 2)` – логиты для классов [норма, ошибка]  
**Требует:** предобученный энкодер (RAE/TransformerAE/VAE)  
**Параметры:** hidden_dim, dropout

**Назначение:** бинарная классификация наличия ошибки в каждом суставе.