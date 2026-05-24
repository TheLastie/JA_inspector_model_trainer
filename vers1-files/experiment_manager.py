import os
import csv
import json
import argparse
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

def create_experiments_csv(csv_path, experiments):
    fieldnames = ['experiment_id', 'exercise', 'status', 'start_time', 'end_time',
                  'best_test_loss', 'best_epoch', 'error_message', 'run_dir']
    param_keys = set()
    for exp in experiments:
        param_keys.update(exp['params'].keys())
    fieldnames.extend(sorted(param_keys))

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            row = {
                'experiment_id': exp['id'],
                'exercise': exp['exercise'],
                'status': 'pending',
            }
            row.update({k: str(v) for k, v in exp['params'].items()})
            writer.writerow(row)

def load_pending_experiments(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row['status'] == 'pending']

def run_experiment(exp_row, args):
    exp_id = exp_row['experiment_id']
    exercise = exp_row['exercise']
    json_file = os.path.join(args.data_dir, f"{exercise}.json")
    output_dir = args.output_base

    # Собираем параметры из CSV строки, пропуская служебные
    cmd_params = []
    for key, value in exp_row.items():
        if key in ['experiment_id', 'exercise', 'status', 'start_time', 'end_time',
                   'best_test_loss', 'best_epoch', 'error_message', 'run_dir']:
            continue
        if value and value != 'None' and value != '':
            # Для флагов типа autoregressive передаём как --autoregressive без значения
            if value.lower() == 'true':
                cmd_params.append(f'--{key}')
            elif value.lower() == 'false':
                cmd_params.append(f'--no_{key}')
            else:
                cmd_params.append(f'--{key}')
                cmd_params.append(str(value))

    cmd = [
        sys.executable, 'run_single_experiment.py',
        '--experiment_id', exp_id,
        '--csv_log', args.csv_log,
        '--json_file', json_file,
        '--output_dir', output_dir,
    ] + cmd_params

    print(f"[{exp_id}] Запуск: {exercise} с параметрами {cmd_params}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[{exp_id}] Ошибка:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    else:
        print(f"[{exp_id}] Успешно завершён.")

def generate_grid(args):
    if not args.exercise:
        print("Укажите --exercise для генерации сетки.")
        return
    experiments = []
    # Определяем сетку параметров
    hidden_sizes = [64, 96, 128]
    seq_lens = [15, 30, 45]
    autoregressive_opts = [True, False]
    for hidden in hidden_sizes:
        for seq_len in seq_lens:
            for ar in autoregressive_opts:
                exp_id = str(uuid.uuid4())[:8]
                experiments.append({
                    'id': exp_id,
                    'exercise': args.exercise,
                    'params': {
                        'hidden_size': hidden,
                        'latent_size': hidden // 2,
                        'seq_len': seq_len,
                        'epochs': args.epochs,
                        'early_stop_patience': args.patience,
                        'autoregressive': ar,
                        'dropout': args.dropout,
                        'lr': args.lr,
                        'batch_size': args.batch_size,
                    }
                })
    create_experiments_csv(args.csv_log, experiments)
    print(f"Сгенерировано {len(experiments)} экспериментов в {args.csv_log}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='datasets')
    parser.add_argument('--output_base', default='models')
    parser.add_argument('--csv_log', default='experiments.csv')
    parser.add_argument('--max_workers', type=int, default=2)
    parser.add_argument('--generate_grid', action='store_true')
    parser.add_argument('--exercise', help='Упражнение для генерации сетки')
    # Параметры для сетки
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    if args.generate_grid:
        generate_grid(args)
        return

    pending = load_pending_experiments(args.csv_log)
    if not pending:
        print("Нет ожидающих экспериментов.")
        return

    print(f"Найдено {len(pending)} экспериментов. Запускаем с {args.max_workers} воркерами.")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_experiment, row, args) for row in pending]
        for future in as_completed(futures):
            future.result()

if __name__ == '__main__':
    main()