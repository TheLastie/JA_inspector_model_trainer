#!/usr/bin/env python3
"""
collect_and_analyze.py – Сбор данных обо всех экспериментах, анализ и генерация спецификации.

Использует ExperimentManager, model_zoo, data_pipeline.
Выполняет:
- Извлечение всех завершённых экспериментов.
- Группировку по упражнениям и типам моделей.
- Построение графиков сравнения точности и зависимости от гиперпараметров.
- Генерацию CSV-отчёта.
- Создание спецификации моделей в формате Markdown (вход/выход, параметры, число параметров).
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
import inspect

from experiment_manager import manager
from model_zoo import RecurrentAutoencoder, TransformerAutoencoder, LSTMVAE, ExerciseAnalyzer

# Настройка стилей
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ----------------------------------------------------------------------
# 1. Сбор данных экспериментов
# ----------------------------------------------------------------------
def collect_experiments_data(filter_model_type=None, filter_exercise=None):
    """Загружает данные всех завершённых экспериментов с best_test_loss."""
    experiments = manager.list_experiments(status='completed')
    records = []
    for exp in experiments:
        if filter_model_type and exp['model_type'] != filter_model_type:
            continue
        if filter_exercise and exp['exercise'] != filter_exercise:
            continue
        if 'best_test_loss' not in exp.get('result', {}):
            continue

        rec = {
            'exp_id': exp['id'],
            'exercise': exp['exercise'],
            'model_type': exp['model_type'],
            'best_test_loss': exp['result']['best_test_loss'],
            'best_epoch': exp['result'].get('best_epoch', -1),
            'created_at': exp['created_at'],
            'finished_at': exp['finished_at'],
        }
        # Добавляем все гиперпараметры
        params = exp.get('params', {})
        for k, v in params.items():
            # Пытаемся преобразовать в число
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                rec[k] = v
        records.append(rec)
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 2. Построение графиков
# ----------------------------------------------------------------------
def plot_best_loss_comparison(df, output_dir='analysis'):
    """Сравнение лучшей ошибки по упражнениям и типам моделей."""
    os.makedirs(output_dir, exist_ok=True)
    # Barplot: средняя ошибка по упражнению и модели
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='exercise', y='best_test_loss', hue='model_type', errorbar='sd')
    plt.title('Best Test Loss by Exercise and Model Type')
    plt.ylabel('Best Test Loss (MSE)')
    plt.xlabel('Exercise')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_loss_comparison.png'), dpi=150)
    plt.close()

    # Boxplot распределения ошибок для каждой модели
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='model_type', y='best_test_loss')
    plt.title('Distribution of Best Test Loss per Model Type')
    plt.ylabel('Best Test Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_distribution_by_model.png'), dpi=150)
    plt.close()


def plot_hyperparameter_dependence(df, params=['hidden_dim', 'latent_dim', 'seq_len', 'dropout', 'lr'],
                                   output_dir='analysis'):
    """Графики зависимости лучшей ошибки от гиперпараметров."""
    os.makedirs(output_dir, exist_ok=True)
    for param in params:
        if param not in df.columns:
            continue
        # Преобразуем в числовой тип, если нужно
        data = df[df[param].notna()].copy()
        if data.empty:
            continue
        # Приводим к float (если строка)
        try:
            data[param] = data[param].astype(float)
        except:
            continue

        plt.figure(figsize=(10, 5))
        if data[param].nunique() < 10:
            # Категориальный / дискретный параметр
            sns.boxplot(data=data, x=param, y='best_test_loss', hue='model_type')
        else:
            sns.scatterplot(data=data, x=param, y='best_test_loss', hue='model_type', alpha=0.7)
            plt.xscale('log') if param == 'lr' else None
        plt.title(f'Best Test Loss vs {param}')
        plt.ylabel('Best Test Loss')
        plt.xlabel(param)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'loss_vs_{param}.png'), dpi=150)
        plt.close()


def plot_heatmap_param_interaction(df, param1='hidden_dim', param2='latent_dim', output_dir='analysis'):
    """Тепловая карта средних ошибок для комбинации двух параметров."""
    os.makedirs(output_dir, exist_ok=True)
    if param1 not in df.columns or param2 not in df.columns:
        print(f"Параметры {param1} или {param2} отсутствуют в данных.")
        return
    # Группировка по среднему
    pivot = df.groupby([param1, param2])['best_test_loss'].mean().reset_index()
    pivot_table = pivot.pivot(index=param1, columns=param2, values='best_test_loss')
    if pivot_table.empty:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis_r', cbar_kws={'label': 'Mean Best Test Loss'})
    plt.title(f'Mean Best Test Loss: {param1} vs {param2}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{param1}_vs_{param2}.png'), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# 3. Генерация спецификации моделей
# ----------------------------------------------------------------------
def count_parameters(model):
    """Подсчёт обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_signature(model_class):
    """Возвращает строку с сигнатурой __init__ (параметры по умолчанию)."""
    sig = inspect.signature(model_class.__init__)
    params = []
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.default != inspect.Parameter.empty:
            params.append(f"{name}={param.default}")
        else:
            params.append(name)
    return f"({', '.join(params)})"


def generate_model_specification(output_file='models_specification.md'):
    """Создаёт Markdown-файл с описанием всех моделей из model_zoo."""
    models_info = {
        'RecurrentAutoencoder': {
            'class': RecurrentAutoencoder,
            'description': 'LSTM-автоэнкодер с авторегрессионным декодером. '
                           'Энкодер сжимает последовательность в латентный вектор, '
                           'декодер восстанавливает последовательность, используя предыдущие предсказания.',
            'input': 'Tensor (batch, seq_len, input_dim) – нормализованные углы (11) или 44 признака.',
            'output': '(reconstructed: Tensor, latent: Tensor). recon имеет ту же форму, что и вход.',
            'typical_params': {
                'input_dim': 11,
                'hidden_dim': 128,
                'latent_dim': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'autoregressive': True
            }
        },
        'TransformerAutoencoder': {
            'class': TransformerAutoencoder,
            'description': 'Трансформерный автоэнкодер. Использует позиционное кодирование, '
                           'энкодер Transformer и декодер Transformer. Latent получается через среднее пулинг.',
            'input': 'Tensor (batch, seq_len, input_dim)',
            'output': '(reconstructed: Tensor, latent: Tensor)',
            'typical_params': {
                'input_dim': 11,
                'd_model': 128,
                'nhead': 4,
                'num_layers': 3,
                'latent_dim': 64,
                'seq_len': 30,
                'dropout': 0.1
            }
        },
        'LSTMVAE': {
            'class': LSTMVAE,
            'description': 'Вариационный LSTM-автоэнкодер. Выдаёт mu и logvar для репараметризации, '
                           'добавляет KL-дивергенцию к MSE.',
            'input': 'Tensor (batch, seq_len, input_dim)',
            'output': '(reconstructed: Tensor, mu: Tensor, logvar: Tensor)',
            'typical_params': {
                'input_dim': 11,
                'hidden_dim': 128,
                'latent_dim': 64,
                'num_layers': 2,
                'dropout': 0.2
            }
        },
        'ExerciseAnalyzer': {
            'class': ExerciseAnalyzer,
            'description': 'Классификатор ошибок по 11 суставам. Требует предобученный энкодер. '
                           'Возвращает список из 11 тензоров логитов формы (batch, 2).',
            'input': 'Tensor (batch, seq_len, input_dim)',
            'output': 'List[Tensor] – каждый тензор (batch, 2) для бинарной классификации сустава.',
            'typical_params': {
                'encoder': 'предобученный RAE/Transformer/VAE',
                'latent_dim': 64,
                'num_joints': 11,
                'hidden_dim': 128,
                'dropout': 0.3
            }
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Спецификация моделей проекта\n\n")
        f.write("## Общая информация\n")
        f.write("- **Входные данные:** Последовательности углов (11 суставов) или расширенные признаки (44).\n")
        f.write("- **Нормализация:** Углы из радиан `[0, π]` преобразуются в `[-1, 1]` по формуле `(angles / π)*2 - 1`.\n")
        f.write("- **Loss для генеративных моделей:** MSE (для VAE добавляется KL-дивергенция с весом 0.001).\n")
        f.write("- **Метрика качества:** `best_test_loss` – MSE на тестовых видео.\n\n")

        for name, info in models_info.items():
            f.write(f"## {name}\n\n")
            f.write(f"**Описание:** {info['description']}\n\n")
            f.write(f"**Сигнатура конструктора:** `{name}{get_model_signature(info['class'])}`\n\n")
            f.write(f"**Вход:** {info['input']}\n\n")
            f.write(f"**Выход:** {info['output']}\n\n")
            f.write("**Типичные гиперпараметры:**\n")
            for param, val in info['typical_params'].items():
                f.write(f"- `{param}`: {val}\n")
            f.write("\n")

            # Подсчёт числа параметров для примера (на стандартных параметрах)
            try:
                # Создаём экземпляр с типичными параметрами, кроме encoder для Analyzer
                if name == 'ExerciseAnalyzer':
                    # Для демонстрации используем фиктивный энкодер
                    class DummyEncoder(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.latent_dim = info['typical_params']['latent_dim']
                        def forward(self, x):
                            return torch.randn(x.size(0), self.latent_dim)
                    model = ExerciseAnalyzer(encoder=DummyEncoder(), **{k:v for k,v in info['typical_params'].items() if k != 'encoder'})
                else:
                    # Убираем input_dim, если он есть, но оставляем для совместимости
                    init_params = {k:v for k,v in info['typical_params'].items() if k != 'input_dim'}
                    model = info['class'](**init_params)
                num_params = count_parameters(model)
                f.write(f"**Число обучаемых параметров (типичная конфигурация):** {num_params:,}\n\n")
            except Exception as e:
                f.write(f"**Число параметров:** не удалось вычислить автоматически ({e})\n\n")

        f.write("\n## Примечания\n")
        f.write("- Все модели ожидают нормализованные входные данные (z-score применяется в `data_pipeline.SequenceDataset`).\n")
        f.write("- `seq_len` – длина окна (обычно 30 кадров).\n")
        f.write("- Для обучения используйте `run_experiment.py` или `massive_trainer.py`.\n")
        f.write("- Результаты экспериментов хранятся в папке `experiments/` и управляются `ExperimentManager`.\n")

    print(f"Спецификация моделей сохранена в {output_file}")


# ----------------------------------------------------------------------
# 4. Основной скрипт
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Сбор и анализ данных экспериментов')
    parser.add_argument('--output_dir', default='analysis', help='Папка для сохранения графиков и отчётов')
    parser.add_argument('--export_csv', default='experiments_summary.csv', help='CSV-файл со сводкой')
    parser.add_argument('--no_plots', action='store_true', help='Не строить графики')
    parser.add_argument('--model_type', help='Фильтр по типу модели (rae, transformer, vae)')
    parser.add_argument('--exercise', help='Фильтр по упражнению')
    args = parser.parse_args()

    # Сбор данных
    df = collect_experiments_data(filter_model_type=args.model_type, filter_exercise=args.exercise)
    if df.empty:
        print("Нет завершённых экспериментов с best_test_loss. Запустите обучение сначала.")
        return

    print(f"Загружено {len(df)} экспериментов.")
    print("\nПервые 5 строк:")
    print(df.head())

    # Сохраняем CSV
    df.to_csv(args.export_csv, index=False)
    print(f"Сводка сохранена в {args.export_csv}")

    # Построение графиков
    if not args.no_plots:
        plot_best_loss_comparison(df, args.output_dir)
        plot_hyperparameter_dependence(df, output_dir=args.output_dir)

        # Дополнительно тепловая карта для популярных пар параметров
        if 'hidden_dim' in df.columns and 'latent_dim' in df.columns:
            plot_heatmap_param_interaction(df, 'hidden_dim', 'latent_dim', args.output_dir)
        if 'seq_len' in df.columns and 'dropout' in df.columns:
            plot_heatmap_param_interaction(df, 'seq_len', 'dropout', args.output_dir)

    # Генерация спецификации моделей
    generate_model_specification('models_specification.md')

    # Краткая статистика
    print("\n=== Статистика по моделям ===")
    stats = df.groupby('model_type')['best_test_loss'].agg(['mean', 'std', 'min', 'count'])
    print(stats)

    print("\n=== Лучшая модель для каждого упражнения ===")
    best_per_exercise = df.loc[df.groupby('exercise')['best_test_loss'].idxmin()]
    print(best_per_exercise[['exercise', 'model_type', 'best_test_loss', 'exp_id']])

if __name__ == '__main__':
    main()