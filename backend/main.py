import sys
import os

# Путь к backend и корню проекта
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from ultralytics import YOLO

from database import init_db
from routers import auth, trainings, settings, stats
import routers.analysis as analysis_mod
from joint_analysis_video import load_ensemble

app = FastAPI(title="FitAnalyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Абсолютные пути
MODELS_ENSEMBLE_DIR = os.path.join(PROJECT_ROOT, "models", "ensemble")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo26x-pose.pt")
STATIC_DIR = os.path.join(BACKEND_DIR, "static")
STATIC_RESULTS_DIR = os.path.join(STATIC_DIR, "results")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

os.makedirs(STATIC_RESULTS_DIR, exist_ok=True)

@app.on_event("startup")
async def startup():
    # Меняем рабочую директорию на корень проекта, чтобы относительные пути
    # внутри joint_analysis_video (models/experiments/..., datasets/...) работали
    os.chdir(PROJECT_ROOT)

    init_db()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
        ensemble_models, classifier, scaler, label_encoder = load_ensemble(
            MODELS_ENSEMBLE_DIR, device, quiet=True
        )
        yolo_model = YOLO(YOLO_MODEL_PATH).to(device)

        # Передаём данные в модуль анализа
        analysis_mod.ensemble_models = ensemble_models
        analysis_mod.classifier = classifier
        analysis_mod.scaler = scaler
        analysis_mod.label_encoder = label_encoder
        analysis_mod.yolo_model = yolo_model
        analysis_mod.device = device
        analysis_mod.thresholds_cache = {}
        analysis_mod.static_results_dir = STATIC_RESULTS_DIR
        analysis_mod.yolo_model_path = YOLO_MODEL_PATH
        print(f"Models loaded. {len(ensemble_models)} ensemble models.")
    except Exception as e:
        print(f"Error loading models: {e}")
        analysis_mod.ensemble_models = {}
        analysis_mod.classifier = None
        analysis_mod.scaler = None
        analysis_mod.label_encoder = None
        analysis_mod.yolo_model = None
        analysis_mod.device = torch.device('cpu')
        analysis_mod.thresholds_cache = {}

# Подключаем роутеры
app.include_router(auth.router)
app.include_router(trainings.router)
app.include_router(settings.router)
app.include_router(stats.router)
app.include_router(analysis_mod.router)

# Статика
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Фронтенд
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)