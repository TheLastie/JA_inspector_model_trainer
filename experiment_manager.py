"""
Модуль experiment_manager.py – управление экспериментами.

Хранит метаданные в JSON, поддерживает:
- Создание эксперимента с уникальным ID и сохранением параметров.
- Обновление статуса и метрик.
- Сохранение артефактов (модели, графики, логи).
- Удаление экспериментов.
- Получение списка всех экспериментов с фильтрацией.
"""

import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any


class ExperimentManager:
    """Менеджер экспериментов с файловым хранилищем."""

    def __init__(self, base_dir: str = "experiments"):
        """
        Args:
            base_dir: корневая папка для хранения всех экспериментов
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_experiment(
        self,
        model_type: str,
        exercise: str,
        params: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Создаёт новый эксперимент.

        Args:
            model_type: тип модели ('rae', 'transformer', 'vae', 'analyzer')
            exercise: название упражнения
            params: словарь гиперпараметров
            tags: опциональные теги для группировки

        Returns:
            exp_id: уникальный идентификатор эксперимента (8 символов)
        """
        exp_id = str(uuid.uuid4())[:8]
        exp_dir = os.path.join(self.base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        meta = {
            "id": exp_id,
            "model_type": model_type,
            "exercise": exercise,
            "params": params,
            "tags": tags or [],
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "finished_at": None,
            "result": {},
            "artifacts": {}
        }
        self._save_meta(exp_dir, meta)
        return exp_id

    def update_status(
        self,
        exp_id: str,
        status: str,
        **metrics
    ) -> None:
        """
        Обновляет статус и добавляет метрики в результат.

        Args:
            exp_id: ID эксперимента
            status: 'running', 'completed', 'failed', 'stopped'
            **metrics: именованные метрики (best_test_loss, best_epoch, accuracy, ...)
        """
        meta = self._load_meta(exp_id)
        if meta is None:
            return

        meta["status"] = status
        now = datetime.now().isoformat()

        if status == "running" and meta["started_at"] is None:
            meta["started_at"] = now
        if status in ("completed", "failed", "stopped") and meta["finished_at"] is None:
            meta["finished_at"] = now

        for key, value in metrics.items():
            meta["result"][key] = value

        exp_dir = os.path.join(self.base_dir, exp_id)
        self._save_meta(exp_dir, meta)

    def save_artifact(
        self,
        exp_id: str,
        artifact_name: str,
        content: bytes,
        artifact_type: str = "file"
    ) -> str:
        """
        Сохраняет артефакт (модель, график, лог) в папку эксперимента.

        Args:
            exp_id: ID эксперимента
            artifact_name: имя файла (например, 'best_model.pth')
            content: бинарное содержимое
            artifact_type: категория ('model', 'plot', 'log')

        Returns:
            полный путь к сохранённому файлу
        """
        exp_dir = os.path.join(self.base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, artifact_name)
        with open(path, 'wb') as f:
            f.write(content)

        meta = self._load_meta(exp_id)
        if meta is not None:
            meta["artifacts"][artifact_name] = {
                "type": artifact_type,
                "path": path,
                "saved_at": datetime.now().isoformat()
            }
            self._save_meta(exp_dir, meta)
        return path

    def copy_artifact(
        self,
        exp_id: str,
        source_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: str = "file"
    ) -> str:
        """Копирует существующий файл в папку эксперимента."""
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if artifact_name is None:
            artifact_name = os.path.basename(source_path)

        exp_dir = os.path.join(self.base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        dest_path = os.path.join(exp_dir, artifact_name)
        shutil.copy2(source_path, dest_path)

        meta = self._load_meta(exp_id)
        if meta is not None:
            meta["artifacts"][artifact_name] = {
                "type": artifact_type,
                "path": dest_path,
                "saved_at": datetime.now().isoformat()
            }
            self._save_meta(exp_dir, meta)
        return dest_path

    def load_experiment(self, exp_id: str) -> Optional[Dict]:
        """Загружает метаданные эксперимента по ID."""
        return self._load_meta(exp_id)

    def list_experiments(
        self,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        exercise: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Возвращает список экспериментов с возможностью фильтрации.

        Args:
            status: фильтр по статусу
            model_type: фильтр по типу модели
            exercise: фильтр по упражнению
            tags: эксперимент должен содержать все указанные теги
        """
        experiments = []
        for exp_id in os.listdir(self.base_dir):
            exp = self._load_meta(exp_id)
            if exp is None:
                continue
            if status is not None and exp.get("status") != status:
                continue
            if model_type is not None and exp.get("model_type") != model_type:
                continue
            if exercise is not None and exp.get("exercise") != exercise:
                continue
            if tags is not None:
                exp_tags = set(exp.get("tags", []))
                if not exp_tags.issuperset(tags):
                    continue
            experiments.append(exp)
        return experiments

    def delete_experiment(self, exp_id: str) -> bool:
        """Удаляет папку эксперимента."""
        exp_dir = os.path.join(self.base_dir, exp_id)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            return True
        return False

    def get_experiment_dir(self, exp_id: str) -> str:
        """Возвращает путь к папке эксперимента."""
        return os.path.join(self.base_dir, exp_id)

    # ---------------------- Внутренние методы ----------------------
    def _load_meta(self, exp_id: str) -> Optional[Dict]:
        meta_path = os.path.join(self.base_dir, exp_id, "meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_meta(self, exp_dir: str, meta: Dict) -> None:
        with open(os.path.join(exp_dir, "meta.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


# Глобальный экземпляр для удобного импорта
manager = ExperimentManager()