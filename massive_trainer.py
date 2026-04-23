#!/usr/bin/env python3
import os
import json
import subprocess
import sys
import threading
import time
import queue
import random
import itertools
import signal
import argparse
import torch
from experiment_manager import manager
from config_manager import load_best_params

DATA_DIR = 'datasets'
MODELS_DIR = 'models'

running_processes = {}
task_queue = queue.Queue()
shutdown_flag = threading.Event()
gpu_cycler = None


def get_next_gpu():
    global gpu_cycler
    if gpu_cycler is None:
        num_gpus = torch.cuda.device_count()
        gpu_cycler = itertools.cycle(range(num_gpus))
        print(f"[GPU] Found {num_gpus} GPUs")
    return next(gpu_cycler)


def worker():
    while not shutdown_flag.is_set():
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        if task is None:
            break
        cmd = task['cmd']
        exp_id = task['exp_id']
        print(f"[Worker] Starting {exp_id}")
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            running_processes[exp_id] = process
            manager.update_status(exp_id, 'running', pid=process.pid)
            process.wait()
            if process.returncode == 0:
                manager.update_status(exp_id, 'completed')
            else:
                stderr = process.stderr.read()
                manager.update_status(exp_id, 'failed', error_message=stderr[:200])
        except Exception as e:
            manager.update_status(exp_id, 'failed', error_message=str(e))
        finally:
            running_processes.pop(exp_id, None)
            task_queue.task_done()


def generate_param_combinations(strategy, param_grid, random_trials):
    """
    Генерирует список словарей с комбинациями гиперпараметров.
    - strategy: 'grid' или 'random'
    - param_grid: словарь вида {'param': [val1, val2, ...]}
    - random_trials: число случайных комбинаций (только для 'random')
    """
    if strategy == 'grid':
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combos]
    else:  # random
        trials = []
        for _ in range(random_trials):
            trial = {k: random.choice(v) for k, v in param_grid.items()}
            trials.append(trial)
        return trials


def run_massive_training(model_type, args, param_combinations=None):
    """
    Запускает обучение для заданного типа модели.
    Если param_combinations передан, то для каждого упражнения и каждой комбинации
    создаётся отдельная задача. Иначе используется одна фиксированная конфигурация.
    """
    print(f"=== Running massive training for {model_type} ===")
    exercises = [f.replace('.json', '') for f in os.listdir(DATA_DIR) if f.endswith('.json')]

    for ex in exercises:
        # Определяем наборы параметров: либо один (базовый), либо несколько (перебор)
        if param_combinations is not None:
            # Перебор: для каждой комбинации создаём задачу
            for idx, combo in enumerate(param_combinations):
                # Объединяем фиксированные параметры из args с перебираемыми
                # Приоритет у значений из combo
                params = {
                    'epochs': args.epochs,
                    'seq_len': args.seq_len,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'hidden_dim': args.hidden_dim,
                    'latent_dim': args.latent_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'early_stop_patience': args.early_stop_patience,
                }
                # Применяем лучшие параметры, если задан флаг
                if args.use_best_config:
                    best = load_best_params(ex, model_type)
                    if best is not None:
                        params.update(best)
                # Переопределяем параметрами из текущей комбинации сетки
                params.update(combo)

                exp_id = manager.create_experiment(model_type, ex, params, tags=['grid_search'])
                gpu_id = get_next_gpu()
                cmd = [
                    sys.executable, 'run_experiment.py',
                    '--model_type', model_type,
                    '--json_file', os.path.join(DATA_DIR, f"{ex}.json"),
                    '--output_dir', MODELS_DIR,
                    '--experiment_id', exp_id,
                    '--gpu_id', str(gpu_id),
                ]
                for k, v in params.items():
                    cmd.append(f'--{k}')
                    cmd.append(str(v))
                if args.use_full_features:
                    cmd.append('--use_full_features')
                if model_type == 'transformer':
                    cmd += ['--d_model', str(args.d_model), '--nhead', str(args.nhead)]

                task_queue.put({'cmd': cmd, 'exp_id': exp_id})
        else:
            # Одна конфигурация (базовая или лучшая)
            params = {
                'epochs': args.epochs,
                'seq_len': args.seq_len,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'hidden_dim': args.hidden_dim,
                'latent_dim': args.latent_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'early_stop_patience': args.early_stop_patience,
            }
            if args.use_best_config:
                best = load_best_params(ex, model_type)
                if best is not None:
                    params.update(best)
                    print(f"Using best config for {ex}_{model_type}: {best}")

            exp_id = manager.create_experiment(model_type, ex, params)
            gpu_id = get_next_gpu()
            cmd = [
                sys.executable, 'run_experiment.py',
                '--model_type', model_type,
                '--json_file', os.path.join(DATA_DIR, f"{ex}.json"),
                '--output_dir', MODELS_DIR,
                '--experiment_id', exp_id,
                '--gpu_id', str(gpu_id),
            ]
            for k, v in params.items():
                cmd.append(f'--{k}')
                cmd.append(str(v))
            if args.use_full_features:
                cmd.append('--use_full_features')
            if model_type == 'transformer':
                cmd += ['--d_model', str(args.d_model), '--nhead', str(args.nhead)]

            task_queue.put({'cmd': cmd, 'exp_id': exp_id})

    print(f"Total tasks in queue: {task_queue.qsize()}")
    task_queue.join()
    print(f"Massive training for {model_type} finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['rae', 'transformer', 'vae', 'all'], default='rae')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--use_full_features', action='store_true')
    parser.add_argument('--use_best_config', action='store_true',
                        help='Use best hyperparameters from previous Optuna runs')
    # Новые аргументы для перебора параметров
    parser.add_argument('--strategy', choices=['grid', 'random'], default=None,
                        help='Hyperparameter search strategy')
    parser.add_argument('--param_grid', type=str, default=None,
                        help='JSON string with parameter grid, e.g. \'{"hidden_dim":[64,128],"latent_dim":[32,64]}\'')
    parser.add_argument('--random_trials', type=int, default=10,
                        help='Number of random trials (only for --strategy random)')
    args = parser.parse_args()

    # Обработка param_grid
    param_combinations = None
    if args.strategy is not None:
        if args.param_grid is None:
            print("Error: --param_grid is required when --strategy is specified.")
            sys.exit(1)
        try:
            param_grid = json.loads(args.param_grid)
        except json.JSONDecodeError as e:
            print(f"Error parsing --param_grid: {e}")
            sys.exit(1)
        param_combinations = generate_param_combinations(args.strategy, param_grid, args.random_trials)
        print(f"Generated {len(param_combinations)} parameter combinations.")

    signal.signal(signal.SIGINT, lambda s, f: shutdown_flag.set())
    for _ in range(args.workers):
        threading.Thread(target=worker).start()

    if args.model_type == 'all':
        for mt in ['rae', 'transformer', 'vae']:
            run_massive_training(mt, args, param_combinations)
    else:
        run_massive_training(args.model_type, args, param_combinations)

    for _ in range(args.workers):
        task_queue.put(None)


if __name__ == '__main__':
    main()