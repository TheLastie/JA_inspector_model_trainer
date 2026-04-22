import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, List, Optional

class ExperimentManager:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_experiment(self, model_type: str, exercise: str, params: Dict) -> str:
        exp_id = str(uuid.uuid4())[:8]
        exp_dir = os.path.join(self.base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        meta = {
            "id": exp_id,
            "model_type": model_type,
            "exercise": exercise,
            "params": params,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "finished_at": None,
            "result": {}
        }
        self._save_meta(exp_dir, meta)
        return exp_id

    def update_status(self, exp_id: str, status: str, **kwargs):
        meta = self.load_experiment(exp_id)
        if meta is None:
            return
        meta["status"] = status
        if status == "running" and meta["started_at"] is None:
            meta["started_at"] = datetime.now().isoformat()
        if status in ("completed", "failed") and meta["finished_at"] is None:
            meta["finished_at"] = datetime.now().isoformat()
        for k, v in kwargs.items():
            meta["result"][k] = v
        exp_dir = os.path.join(self.base_dir, exp_id)
        self._save_meta(exp_dir, meta)

    def load_experiment(self, exp_id: str) -> Optional[Dict]:
        meta_path = os.path.join(self.base_dir, exp_id, "meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_experiments(self, status: Optional[str] = None) -> List[Dict]:
        experiments = []
        for exp_id in os.listdir(self.base_dir):
            exp = self.load_experiment(exp_id)
            if exp and (status is None or exp.get("status") == status):
                experiments.append(exp)
        return experiments

    def delete_experiment(self, exp_id: str):
        exp_dir = os.path.join(self.base_dir, exp_id)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)

    def get_experiment_dir(self, exp_id: str) -> str:
        return os.path.join(self.base_dir, exp_id)

    def _save_meta(self, exp_dir: str, meta: Dict):
        with open(os.path.join(exp_dir, "meta.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

manager = ExperimentManager()