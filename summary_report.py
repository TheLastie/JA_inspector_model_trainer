#!/usr/bin/env python3
"""
Генерация сводного отчёта по всем экспериментам.
Выводит топ-2 модели для каждого упражнения и сохраняет CSV.
"""

import os
import csv
from collections import defaultdict
from experiment_manager import manager


def get_top_models(top_n: int = 2):
    """Возвращает список лучших моделей для каждого упражнения."""
    completed = manager.list_experiments(status='completed')
    # Оставляем только те, где есть best_test_loss
    valid = [e for e in completed if 'best_test_loss' in e['result']]

    by_exercise = defaultdict(list)
    for exp in valid:
        by_exercise[exp['exercise']].append(exp)

    report = []
    for exercise in sorted(by_exercise.keys()):
        sorted_exps = sorted(by_exercise[exercise], key=lambda e: e['result']['best_test_loss'])
        for i, exp in enumerate(sorted_exps[:top_n], 1):
            report.append({
                'Exercise': exercise,
                'Rank': i,
                'Model': exp['model_type'],
                'Test Loss': f"{exp['result']['best_test_loss']:.6f}",
                'Epoch': exp['result'].get('best_epoch', ''),
                'Exp ID': exp['id'],
                'Params': ', '.join(f"{k}={v}" for k, v in exp['params'].items()
                                    if k in ['hidden_dim', 'latent_dim', 'seq_len', 'lr', 'dropout'])
            })
    return report


def main():
    report = get_top_models(top_n=2)

    if not report:
        print("No completed experiments with best_test_loss found.")
        return

    # Вывод в консоль
    print(f"{'Exercise':<20} {'Rank':<5} {'Model':<12} {'Test Loss':<12} {'Epoch':<6} {'Exp ID':<10} {'Params'}")
    print('-' * 100)
    for r in report:
        print(f"{r['Exercise']:<20} {r['Rank']:<5} {r['Model']:<12} {r['Test Loss']:<12} "
              f"{r['Epoch']:<6} {r['Exp ID']:<10} {r['Params']}")

    # Сохранение CSV
    with open('top_models_report.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    print("\nReport saved to top_models_report.csv")


if __name__ == '__main__':
    main()