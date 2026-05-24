#!/usr/bin/env python3
import os
import re
import csv
from collections import defaultdict

MODELS_DIR = 'models'

def extract_metrics_from_log(log_path):
    """Извлекает best_test_loss и best_epoch из training_log.txt."""
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, 'r') as f:
        content = f.read()
    # Формат: "Best epoch: 26 | Test Loss: 0.002952"
    match = re.search(r'Best epoch:\s*(\d+)\s*\|\s*Test Loss:\s*([\d\.]+)', content)
    if match:
        return int(match.group(1)), float(match.group(2))
    return None, None

def get_top_models(top_n=2):
    models = []
    for exercise in os.listdir(MODELS_DIR):
        ex_path = os.path.join(MODELS_DIR, exercise)
        if not os.path.isdir(ex_path):
            continue
        for run in os.listdir(ex_path):
            run_path = os.path.join(ex_path, run)
            if not run.startswith('run_'):
                continue
            log_file = os.path.join(run_path, 'training_log.txt')
            epoch, loss = extract_metrics_from_log(log_file)
            if loss is None:
                continue
            models.append({
                'exercise': exercise,
                'run': run,
                'test_loss': loss,
                'epoch': epoch,
                'run_dir': run_path
            })
    
    # Группируем и выбираем лучшие
    by_exercise = defaultdict(list)
    for m in models:
        by_exercise[m['exercise']].append(m)
    
    report = []
    for ex in sorted(by_exercise.keys()):
        sorted_models = sorted(by_exercise[ex], key=lambda x: x['test_loss'])
        for i, m in enumerate(sorted_models[:top_n], 1):
            report.append({
                'Exercise': ex,
                'Rank': i,
                'Test Loss': f"{m['test_loss']:.6f}",
                'Epoch': m['epoch'],
                'Run': m['run']
            })
    return report

if __name__ == '__main__':
    report = get_top_models(top_n=2)
    if not report:
        print("Не найдено обученных моделей с training_log.txt.")
    else:
        # Вывод в консоль
        print(f"{'Exercise':<20} {'Rank':<5} {'Test Loss':<12} {'Epoch':<6} {'Run'}")
        print('-' * 60)
        for r in report:
            print(f"{r['Exercise']:<20} {r['Rank']:<5} {r['Test Loss']:<12} {r['Epoch']:<6} {r['Run']}")
        
        # Сохранение CSV
        with open('top_models_report.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Exercise', 'Rank', 'Test Loss', 'Epoch', 'Run'])
            writer.writeheader()
            writer.writerows(report)
        print("\nОтчёт сохранён в top_models_report.csv")