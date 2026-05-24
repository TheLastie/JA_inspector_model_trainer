#!/usr/bin/env python3
"""
visualize_experiments.py – Визуализация результатов экспериментов из папки experiments.

Строит графики зависимости ошибки (best_test_loss) от гиперпараметров
для всех завершённых экспериментов, сохранённых через ExperimentManager.
"""

import os
import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from experiment_manager import manager

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_experiments_data(filter_model_type=None, filter_exercise=None):
    """
    Загружает данные всех завершённых экспериментов с наличием best_test_loss.
    Возвращает список словарей с полями:
        - exercise
        - model_type
        - best_test_loss
        - все параметры из params
    """
    experiments = manager.list_experiments(status='completed')
    data = []
    for exp in experiments:
        # Фильтры
        if filter_model_type and exp['model_type'] != filter_model_type:
            continue
        if filter_exercise and exp['exercise'] != filter_exercise:
            continue
        if 'best_test_loss' not in exp.get('result', {}):
            continue

        entry = {
            'exercise': exp['exercise'],
            'model_type': exp['model_type'],
            'best_test_loss': exp['result']['best_test_loss'],
            'exp_id': exp['id'],
        }
        # Добавляем все параметры
        params = exp.get('params', {})
        for k, v in params.items():
            # Приводим к числу, если возможно
            try:
                entry[k] = float(v)
            except (TypeError, ValueError):
                entry[k] = v
        data.append(entry)
    return data


def plot_scatter_params(df, params_to_plot, hue='model_type', output_dir='plots'):
    """
    Строит scatter plots для каждого параметра из params_to_plot
    (зависимость best_test_loss от значения параметра).
    """
    os.makedirs(output_dir, exist_ok=True)
    for param in params_to_plot:
        if param not in df.columns:
            print(f"Параметр '{param}' отсутствует в данных, пропускаем.")
            continue

        plt.figure()
        # Используем stripplot или scatter с небольшим jitter для дискретных значений
        if df[param].nunique() < 10:
            sns.stripplot(data=df, x=param, y='best_test_loss', hue=hue, dodge=True, alpha=0.7)
        else:
            sns.scatterplot(data=df, x=param, y='best_test_loss', hue=hue, alpha=0.7)
        plt.title(f'Best Test Loss vs {param}')
        plt.ylabel('Best Test Loss')
        plt.xlabel(param)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'scatter_{param}.png'), dpi=150)
        plt.close()
        print(f"Сохранён scatter plot для {param}")


def plot_pair_heatmap(df, param_pairs, hue='model_type', output_dir='plots'):
    """
    Для каждой пары параметров строит heatmap среднего значения best_test_loss.
    param_pairs - список кортежей (p1, p2)
    """
    os.makedirs(output_dir, exist_ok=True)
    for p1, p2 in param_pairs:
        if p1 not in df.columns or p2 not in df.columns:
            print(f"Пара ({p1}, {p2}) содержит отсутствующий параметр, пропускаем.")
            continue

        # Группируем по паре параметров и считаем средний loss
        pivot = df.groupby([p1, p2])['best_test_loss'].mean().reset_index()
        pivot_table = pivot.pivot(index=p1, columns=p2, values='best_test_loss')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis_r', cbar_kws={'label': 'Mean Best Test Loss'})
        plt.title(f'Mean Best Test Loss: {p1} vs {p2}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{p1}_vs_{p2}.png'), dpi=150)
        plt.close()
        print(f"Сохранён heatmap для {p1} vs {p2}")


def plot_boxen_by_model(df, param, output_dir='plots'):
    """Строит boxen plot распределения ошибки по значениям параметра для разных моделей."""
    if param not in df.columns:
        return
    plt.figure(figsize=(12, 6))
    sns.boxenplot(data=df, x=param, y='best_test_loss', hue='model_type')
    plt.title(f'Distribution of Best Test Loss by {param} and Model Type')
    plt.ylabel('Best Test Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxen_{param}_by_model.png'), dpi=150)
    plt.close()
    print(f"Сохранён boxen plot для {param}")


def parse_pairs(pair_strings):
    """
    Преобразует список строк вида 'p1,p2' или 'p1 p2' в список кортежей (p1, p2).
    """
    pairs = []
    for s in pair_strings:
        # Поддерживаем как запятую, так и пробел в качестве разделителя
        if ',' in s:
            parts = s.split(',')
        else:
            parts = s.split()
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
        else:
            print(f"Пропущена некорректная пара: '{s}' (должно быть два параметра через запятую или пробел)")
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Визуализация зависимости ошибки от гиперпараметров')
    parser.add_argument('--model_type', choices=['rae', 'transformer', 'vae', 'analyzer'],
                        help='Фильтр по типу модели')
    parser.add_argument('--exercise', help='Фильтр по упражнению (например, squat)')
    parser.add_argument('--output_dir', default='plots', help='Папка для сохранения графиков')
    parser.add_argument('--params', nargs='+',
                        default=['hidden_dim', 'latent_dim', 'seq_len', 'dropout', 'lr', 'num_layers'],
                        help='Список параметров для визуализации (scatter)')
    parser.add_argument('--pairs', nargs='+',
                        default=['hidden_dim,latent_dim', 'seq_len,dropout'],
                        help='Пары параметров для heatmap в формате "p1,p2" (можно несколько через пробел)')
    args = parser.parse_args()

    # Обработка пар параметров
    pairs = parse_pairs(args.pairs)

    # Загрузка данных
    df = pd.DataFrame(load_experiments_data(args.model_type, args.exercise))
    if df.empty:
        print("Нет завершённых экспериментов с best_test_loss, удовлетворяющих фильтрам.")
        return

    print(f"Загружено {len(df)} экспериментов.")
    print("Статистика по best_test_loss:")
    print(df['best_test_loss'].describe())

    # Scatter plots для отдельных параметров
    plot_scatter_params(df, args.params, output_dir=args.output_dir)

    # Heatmap для пар параметров
    if pairs:
        plot_pair_heatmap(df, pairs, output_dir=args.output_dir)

    # Дополнительно: boxen для каждого параметра с разбивкой по model_type
    for param in args.params:
        if param in df.columns and df[param].nunique() <= 10:
            plot_boxen_by_model(df, param, args.output_dir)

    print("Готово.")


if __name__ == '__main__':
    main()