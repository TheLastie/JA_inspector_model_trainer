#!/usr/bin/env python3
import os
import json
import csv
import pandas as pd
from glob import glob

MODELS_DIR = 'models'
ANALYZERS_DIR = 'analyzers'
OUTPUT_CSV = 'model_comparison.csv'
OUTPUT_HTML = 'model_comparison.html'

def collect_results():
    records = []
    # RAE модели
    for log_path in glob(f'{MODELS_DIR}/*/run_*/training_log.txt'):
        rec = parse_rae_log(log_path)
        if rec:
            rec['model_type'] = 'RAE'
            records.append(rec)
    # Analyzer модели
    for log_path in glob(f'{ANALYZERS_DIR}/*/analyzer_*/training_log.txt'):
        rec = parse_analyzer_log(log_path)
        if rec:
            rec['model_type'] = 'Analyzer'
            records.append(rec)
    # ST-GCN (предположим, что они сохраняются в models_stgcn)
    for log_path in glob(f'models_stgcn/*/run_*/training_log.txt'):
        rec = parse_generic_log(log_path)
        rec['model_type'] = 'ST-GCN'
        records.append(rec)
    # Transformer, VAE, OCSVM аналогично
    # ...
    return records

def parse_rae_log(path):
    with open(path) as f:
        content = f.read()
    # извлечь best_test_loss и best_epoch
    loss = None
    for line in content.split('\n'):
        if 'Best epoch:' in line:
            parts = line.split('|')
            loss = float(parts[1].split(':')[1].strip())
            break
    if loss is None:
        return None
    return {'exercise': os.path.basename(os.path.dirname(os.path.dirname(path))),
            'run': os.path.basename(os.path.dirname(path)),
            'test_loss': loss}

def generate_comparison_table(records):
    df = pd.DataFrame(records)
    # группируем по упражнению и типу модели, находим лучшую по test_loss
    best_per_exercise = df.loc[df.groupby(['exercise', 'model_type'])['test_loss'].idxmin()]
    best_per_exercise = best_per_exercise.sort_values(['exercise', 'test_loss'])
    best_per_exercise.to_csv(OUTPUT_CSV, index=False)
    best_per_exercise.to_html(OUTPUT_HTML, index=False)
    print(f"Comparison saved to {OUTPUT_CSV} and {OUTPUT_HTML}")

if __name__ == '__main__':
    records = collect_results()
    if records:
        generate_comparison_table(records)
    else:
        print("No results found.")