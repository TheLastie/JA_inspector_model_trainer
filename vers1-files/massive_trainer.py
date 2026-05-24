#!/usr/bin/env python3
import os
import json
import subprocess
import sys
import uuid
import threading
import time
import queue
import random
import itertools
import signal
import torch
from datetime import datetime
from exp_manager2 import manager

DATA_DIR = 'datasets'
MODELS_DIR = 'models'
ANALYZERS_DIR = 'analyzers'

running_processes = {}
task_queue = queue.Queue()
MAX_WORKERS = 2
shutdown_flag = threading.Event()

gpu_cycler = None

def get_next_gpu():
    global gpu_cycler
    if gpu_cycler is None:
        num_gpus = torch.cuda.device_count()
        gpu_cycler = itertools.cycle(range(num_gpus))
        print(f"[GPU] Found {num_gpus} GPUs, will distribute tasks")
    return next(gpu_cycler)

def get_available_models(model_type='rae'):
    base_dir = MODELS_DIR if model_type == 'rae' else ANALYZERS_DIR
    models = []
    if not os.path.exists(base_dir):
        return models
    for exercise in os.listdir(base_dir):
        ex_path = os.path.join(base_dir, exercise)
        if not os.path.isdir(ex_path):
            continue
        for run in os.listdir(ex_path):
            run_path = os.path.join(ex_path, run)
            if (model_type == 'rae' and run.startswith('run_')) or (model_type == 'analyzer' and run.startswith('analyzer_')):
                model_file = os.path.join(run_path, 'best_model.pth' if model_type == 'rae' else 'best_analyzer.pth')
                if os.path.exists(model_file):
                    models.append({'exercise': exercise, 'model_path': model_file})
    return models

def start_training_thread(task_type, exercise, params, script):
    exp_id = manager.create_experiment(task_type, exercise, params)
    json_file = os.path.join(DATA_DIR, f"{exercise}.json")
    if not os.path.exists(json_file):
        manager.update_status(exp_id, "failed", error_message="JSON not found")
        return None

    cmd_params = []
    for k, v in params.items():
        if v is not None and v != '':
            cmd_params.append(f'--{k}')
            cmd_params.append(str(v))

    gpu_id = get_next_gpu()
    cmd_params.append('--gpu_id')
    cmd_params.append(str(gpu_id))

    if task_type == 'rae':
        cmd = [sys.executable, script, '--json_file', json_file, '--output_base', MODELS_DIR,
               '--experiment_id', exp_id] + cmd_params
    else:
        rae_models = [m for m in get_available_models('rae') if m['exercise'] == exercise]
        if not rae_models:
            manager.update_status(exp_id, "failed", error_message="No RAE model found")
            return None
        latest_rae = rae_models[0]['model_path']
        cmd = [sys.executable, script, '--json_file', json_file, '--rae_model', latest_rae,
               '--output_base', ANALYZERS_DIR, '--experiment_id', exp_id] + cmd_params

    print(f"[Process] Starting on GPU {gpu_id}: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        manager.update_status(exp_id, "failed", error_message=str(e))
        return None

    running_processes[exp_id] = process
    manager.update_status(exp_id, "running", pid=process.pid)

    def monitor():
        process.wait()
        end_time = datetime.now().isoformat()
        if process.returncode == 0:
            manager.update_status(exp_id, "completed")
            print(f"[Monitor] {exp_id} completed")
        else:
            stderr = process.stderr.read()
            # Сохраняем ошибку в папку эксперимента
            exp_dir = manager.get_experiment_dir(exp_id)
            with open(os.path.join(exp_dir, "error.log"), 'w') as f:
                f.write(stderr)
            manager.update_status(exp_id, "failed", error_message=stderr[:200])
            print(f"[Monitor] {exp_id} failed, log: {exp_dir}/error.log")
        running_processes.pop(exp_id, None)

    threading.Thread(target=monitor).start()
    return exp_id

def generate_param_combinations(strategy, param_grid, random_trials=10):
    if strategy == 'grid':
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    else:
        trials = []
        for _ in range(random_trials):
            trial = {}
            for key, vals in param_grid.items():
                trial[key] = random.choice(vals)
            trials.append(trial)
        return trials

def worker(script):
    while not shutdown_flag.is_set():
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            continue
        if task is None:
            break
        task_type = task['type']
        exercise = task['exercise']
        params = task['params']
        print(f"[Worker] {task_type} on {exercise}")
        exp_id = start_training_thread(task_type, exercise, params, script)
        if exp_id:
            process = running_processes.get(exp_id)
            if process:
                try:
                    while process.poll() is None and not shutdown_flag.is_set():
                        time.sleep(1)
                except:
                    pass
        task_queue.task_done()

def signal_handler(sig, frame):
    print("\n[Massive] Interrupted, shutting down...")
    shutdown_flag.set()
    while not task_queue.empty():
        try:
            task_queue.get_nowait()
            task_queue.task_done()
        except:
            pass
    sys.exit(0)

def run_massive_training(model_type, args):
    if model_type == 'rae':
        script = 'train_rae_per_json.py'
        default_grid = {
            'hidden_size': [64, 96, 128],
            'latent_size': [32, 48, 64],
            'seq_len': [15, 30, 45],
            'lr': [0.001, 0.005, 0.01],
            'dropout': [0.1, 0.2, 0.3]
        }
    else:
        script = 'train_exercise_analyzer.py'
        default_grid = {
            'analyzer_hidden': [64, 128],
            'freeze_encoder': [True, False],
            'lr': [0.001, 0.0005],
            'epochs': [30, 50]
        }

    param_grid = default_grid
    if args.param_grid:
        try:
            param_grid = json.loads(args.param_grid)
        except:
            print("Invalid param_grid JSON, using default.")

    print(f"=== Running massive training for {model_type} ===")
    print(f"Strategy: {args.strategy}, trials: {args.random_trials}")
    print(f"Parameter grid: {param_grid}")

    exercises = [f.replace('.json', '') for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"Found {len(exercises)} exercises")

    for _ in range(args.workers):
        t = threading.Thread(target=worker, args=(script,))
        t.start()

    for ex in exercises:
        if model_type == 'analyzer':
            rae_models = get_available_models('rae')
            if not any(m['exercise'] == ex for m in rae_models):
                print(f"Skipping {ex}: no RAE model available")
                continue
        param_combinations = generate_param_combinations(args.strategy, param_grid, args.random_trials)
        for params in param_combinations:
            task_queue.put({'type': model_type, 'exercise': ex, 'params': params})

    print(f"Total tasks: {task_queue.qsize()}")
    task_queue.join()

    for _ in range(args.workers):
        task_queue.put(None)
    print("Massive training finished.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['rae', 'analyzer', 'all'], default='rae')
    parser.add_argument('--strategy', choices=['grid', 'random'], default='grid')
    parser.add_argument('--random_trials', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--param_grid', type=str, help='JSON string with parameter grid')
    args = parser.parse_args()

    global MAX_WORKERS
    MAX_WORKERS = args.workers

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.model_type == 'all':
        for mtype in ['rae', 'analyzer']:
            if shutdown_flag.is_set():
                break
            run_massive_training(mtype, args)
    else:
        run_massive_training(args.model_type, args)

if __name__ == '__main__':
    main()