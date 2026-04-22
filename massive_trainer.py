#!/usr/bin/env python3
"""
massive_trainer.py – оркестратор массового обучения с перебором гиперпараметров.
"""

import os
import sys
import json
import argparse
import itertools
import subprocess
import threading
import queue
import time
import signal
import random
import torch
from datetime import datetime
from typing import Dict, List, Optional

from experiment_manager import manager

# Конфигурация
DATA_DIR = 'datasets'
OUTPUT_DIR = 'models'
ANALYZER_OUTPUT_DIR = 'analyzers'

shutdown_flag = threading.Event()
task_queue = queue.Queue()
running_processes: Dict[str, subprocess.Popen] = {}
gpu_cycler = None

def get_next_gpu():
    global gpu_cycler
    if gpu_cycler is None:
        num_gpus = torch.cuda.device_count()
        gpu_cycler = itertools.cycle(range(num_gpus))
        print(f"[GPU] Found {num_gpus} GPUs, will distribute tasks")
    return next(gpu_cycler)

def generate_param_combinations(strategy: str, param_grid: Dict, random_trials: int = 10) -> List[Dict]:
    if strategy == 'grid':
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    else:  # random
        trials = []
        for _ in range(random_trials):
            trial = {}
            for key, vals in param_grid.items():
                trial[key] = random.choice(vals)
            trials.append(trial)
        return trials

def worker(script: str, fixed_params: Dict):
    while not shutdown_flag.is_set():
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        if task is None:
            break

        exercise = task['exercise']
        params = {**fixed_params, **task['params']}  # фиксированные параметры переопределяют

        # Создаём эксперимент
        exp_id = manager.create_experiment(
            model_type=task['model_type'],
            exercise=exercise,
            params=params
        )
        json_file = os.path.join(DATA_DIR, f"{exercise}.json")
        if not os.path.exists(json_file):
            manager.update_status(exp_id, 'failed', error_message='JSON not found')
            task_queue.task_done()
            continue

        # Собираем аргументы командной строки
        cmd = [
            sys.executable, script,
            '--model_type', task['model_type'],
            '--json_file', json_file,
            '--output_dir', OUTPUT_DIR,
            '--experiment_id', exp_id,
            '--gpu_id', str(get_next_gpu())
        ]
        for key, value in params.items():
            if value is not None and value != '':
                cmd.append(f'--{key}')
                cmd.append(str(value))

        print(f"[Process] Starting: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as e:
            manager.update_status(exp_id, 'failed', error_message=str(e))
            task_queue.task_done()
            continue

        running_processes[exp_id] = proc
        manager.update_status(exp_id, 'running', pid=proc.pid)

        # Ожидаем завершения
        proc.wait()
        if proc.returncode == 0:
            manager.update_status(exp_id, 'completed')
        else:
            stderr = proc.stderr.read()
            manager.update_status(exp_id, 'failed', error_message=stderr[:200])
        running_processes.pop(exp_id, None)
        task_queue.task_done()

def run_massive_training(model_type: str, args):
    script = 'run_experiment.py'
    default_grid = {
        'rae': {
            'hidden_dim': [64, 96, 128],
            'latent_dim': [32, 48, 64],
            'seq_len': [15, 30, 45],
            'lr': [0.001, 0.005, 0.01],
            'dropout': [0.1, 0.2, 0.3]
        },
        'transformer': {
            'd_model': [64, 128],
            'nhead': [4, 8],
            'num_layers': [2, 3],
            'latent_dim': [32, 64],
            'seq_len': [15, 30, 45]
        },
        'vae': {
            'hidden_dim': [64, 128],
            'latent_dim': [32, 64],
            'num_layers': [1, 2],
            'seq_len': [15, 30, 45]
        }
    }.get(model_type, {})

    param_grid = default_grid
    if args.param_grid:
        try:
            param_grid = json.loads(args.param_grid)
        except:
            print("Invalid param_grid JSON, using default.")

    fixed_params = {}
    if args.epochs:
        fixed_params['epochs'] = args.epochs
    if args.batch_size:
        fixed_params['batch_size'] = args.batch_size

    print(f"=== Running massive training for {model_type} ===")
    print(f"Strategy: {args.strategy}, trials: {args.random_trials}")
    print(f"Parameter grid: {param_grid}")
    print(f"Fixed parameters: {fixed_params}")

    exercises = [f.replace('.json', '') for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"Found {len(exercises)} exercises")

    # Запускаем воркеры
    for _ in range(args.workers):
        t = threading.Thread(target=worker, args=(script, fixed_params))
        t.start()

    # Наполняем очередь
    for ex in exercises:
        param_combinations = generate_param_combinations(args.strategy, param_grid, args.random_trials)
        for params in param_combinations:
            task_queue.put({
                'model_type': model_type,
                'exercise': ex,
                'params': params
            })

    print(f"Total tasks: {task_queue.qsize()}")
    task_queue.join()

    # Останавливаем воркеры
    for _ in range(args.workers):
        task_queue.put(None)
    print(f"Massive training for {model_type} finished.")

def signal_handler(sig, frame):
    print("\n[Massive] Interrupted, shutting down...")
    shutdown_flag.set()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['rae', 'transformer', 'vae', 'all'], default='rae')
    parser.add_argument('--strategy', choices=['grid', 'random'], default='grid')
    parser.add_argument('--random_trials', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--param_grid', type=str, help='JSON with parameter grid')
    parser.add_argument('--epochs', type=int, help='Fixed number of epochs for all experiments')
    parser.add_argument('--batch_size', type=int, help='Fixed batch size for all experiments')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.model_type == 'all':
        for mtype in ['rae', 'transformer', 'vae']:
            if shutdown_flag.is_set():
                break
            run_massive_training(mtype, args)
    else:
        run_massive_training(args.model_type, args)

if __name__ == '__main__':
    main()