#!/usr/bin/env python3
"""
Подбор гиперпараметров с помощью Optuna.
Запускает run_experiment.py с разными параметрами, сохраняет результаты в ExperimentManager.
"""

import os
import sys
import json
import subprocess
import argparse
from typing import Dict, Any
import optuna
from optuna.trial import Trial

from experiment_manager import manager


def objective(trial: Trial, base_args: Dict[str, Any]) -> float:
    """Целевая функция для Optuna."""
    # Генерируем гиперпараметры
    params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 96, 128]),
        'latent_dim': trial.suggest_categorical('latent_dim', [32, 48, 64]),
        'seq_len': trial.suggest_categorical('seq_len', [15, 30, 45]),
        'lr': trial.suggest_categorical('lr', [0.001, 0.005, 0.01]),
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'batch_size': 32,
        'epochs': 50,  # ограниченное число для поиска
        'early_stop_patience': 10,
    }

    # Создаём эксперимент в менеджере
    exp_id = manager.create_experiment(
        model_type=base_args['model_type'],
        exercise=base_args['exercise'],
        params=params,
        tags=['optuna', f"trial_{trial.number}"]
    )
    manager.update_status(exp_id, 'pending')

    # Формируем команду
    cmd = [
        sys.executable, 'run_experiment.py',
        '--model_type', base_args['model_type'],
        '--json_file', base_args['json_file'],
        '--output_dir', base_args['output_dir'],
        '--experiment_id', exp_id,
    ]
    for k, v in params.items():
        cmd.append(f'--{k}')
        cmd.append(str(v))
    if base_args.get('gpu_id') is not None:
        cmd.extend(['--gpu_id', str(base_args['gpu_id'])])

    # Запускаем
    print(f"Trial {trial.number}: {params}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Проверяем результат
    exp = manager.load_experiment(exp_id)
    if exp and exp['status'] == 'completed' and 'best_test_loss' in exp['result']:
        loss = exp['result']['best_test_loss']
        trial.set_user_attr('exp_id', exp_id)
        return loss
    else:
        # Неудача – возвращаем большое значение
        return 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['rae', 'transformer', 'vae'])
    parser.add_argument('--json_file', required=True)
    parser.add_argument('--output_dir', default='models')
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    exercise = os.path.splitext(os.path.basename(args.json_file))[0]
    base_args = {
        'model_type': args.model_type,
        'json_file': args.json_file,
        'output_dir': args.output_dir,
        'exercise': exercise,
        'gpu_id': args.gpu_id,
    }

    study = optuna.create_study(
        direction='minimize',
        study_name=f"{exercise}_{args.model_type}",
        storage=f"sqlite:///optuna_{exercise}_{args.model_type}.db",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, base_args), n_trials=args.n_trials)

    print("\n=== Best trial ===")
    print(f"Value: {study.best_value:.6f}")
    print("Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Experiment ID: {study.best_trial.user_attrs.get('exp_id', 'unknown')}")


if __name__ == '__main__':
    main()