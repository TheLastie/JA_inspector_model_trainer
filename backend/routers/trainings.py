from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime

from schemas import TrainingCreate, TrainingOut
from models import Training, MuscleStats, User
from auth import get_current_user, get_db

router = APIRouter(prefix="/api/trainings", tags=["trainings"])

MUSCLE_MAP = {
    "Приседания": ["legs", "core"],
    "Отжимания": ["arms", "core"],
    "Планка": ["core", "arms"],
    "Выпады": ["legs", "core"],
    "Подтягивания": ["arms", "core"],
}

@router.get("/", response_model=List[TrainingOut])
def get_trainings(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trainings = db.query(Training).filter(Training.user_id == user.id).order_by(Training.date.desc()).all()
    return [
        TrainingOut(
            id=t.id,
            exerciseName=t.exercise_name,
            date=t.date,
            reps=t.reps,
            errors=t.errors,
            duration=t.duration,
            score_10=t.score_10,
            score_5=t.score_5,
            joint_errors=t.joint_errors,
            segments=t.segments,
            video_url=t.video_url
        ) for t in trainings
    ]

@router.post("/", response_model=TrainingOut, status_code=201)
def create_training(
    data: TrainingCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    training = Training(
        id=str(uuid.uuid4()),
        user_id=user.id,
        exercise_name=data.exerciseName,
        date=data.date,
        reps=data.reps,
        errors=data.errors,
        duration=data.duration,
        score_10=data.score_10,
        score_5=data.score_5,
        joint_errors=data.joint_errors,
        segments=data.segments,
        video_url=data.video_url
    )
    db.add(training)

    muscle_groups = MUSCLE_MAP.get(data.exerciseName, [])
    if muscle_groups:
        stats = db.query(MuscleStats).filter(MuscleStats.user_id == user.id).first()
        if not stats:
            stats = MuscleStats(user_id=user.id)
            db.add(stats)
        for group in muscle_groups:
            setattr(stats, group, getattr(stats, group) + data.reps)
        stats.total += data.reps

    db.commit()
    db.refresh(training)

    return TrainingOut(
        id=training.id,
        exerciseName=training.exercise_name,
        date=training.date,
        reps=training.reps,
        errors=training.errors,
        duration=training.duration,
        score_10=training.score_10,
        score_5=training.score_5,
        joint_errors=training.joint_errors,
        segments=training.segments,
        video_url=training.video_url
    )

@router.delete("/{training_id}")
def delete_training(
    training_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    training = db.query(Training).filter(Training.id == training_id, Training.user_id == user.id).first()
    if not training:
        raise HTTPException(status_code=404, detail="Тренировка не найдена")
    db.delete(training)
    db.commit()
    return {"success": True}