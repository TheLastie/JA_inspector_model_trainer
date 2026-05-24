"""
Модуль для сохранения и загрузки лучших гиперпараметров,
найденных Optuna, для каждой пары (упражнение, тип модели).
"""

import json
import os
from typing import Dict, Any, Optional

CONFIG_FILE = "best_hyperparams.json"


def save_best_params(exercise: str, model_type: str, params: Dict[str, Any]) -> None:
    """
    Сохраняет лучшие параметры для заданного упражнения и типа модели.
    
    Args:
        exercise: название упражнения (без расширения)
        model_type: 'rae', 'transformer', 'vae'
        params: словарь с гиперпараметрами
    """
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)

    key = f"{exercise}_{model_type}"
    config[key] = params

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_best_params(exercise: str, model_type: str) -> Optional[Dict[str, Any]]:
    """
    Загружает лучшие параметры для пары упражнение-модель.
    
    Returns:
        Словарь параметров или None, если запись не найдена.
    """
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    key = f"{exercise}_{model_type}"
    return config.get(key)