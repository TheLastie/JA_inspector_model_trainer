from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta

from models import MuscleStats, Training
from auth import get_current_user, get_db
from schemas import MuscleStatsOut, DailyReps

router = APIRouter(prefix="/api/stats", tags=["stats"])

@router.get("/muscles", response_model=MuscleStatsOut)
def muscle_stats(user=Depends(get_current_user), db: Session = Depends(get_db)):
    stats = db.query(MuscleStats).filter(MuscleStats.user_id == user.id).first()
    if not stats:
        return MuscleStatsOut(legs=0, core=0, arms=0, total=0)
    return MuscleStatsOut(
        legs=stats.legs,
        core=stats.core,
        arms=stats.arms,
        total=stats.total
    )

@router.get("/weekly", response_model=List[DailyReps])
def weekly_stats(user=Depends(get_current_user), db: Session = Depends(get_db)):
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    start_str = seven_days_ago.isoformat()

    trainings = db.query(Training).filter(
        Training.user_id == user.id,
        Training.date >= start_str
    ).order_by(Training.date.asc()).all()

    # Группируем по дням (YYYY-MM-DD)
    daily = {}
    for t in trainings:
        day = t.date[:10]
        daily[day] = daily.get(day, 0) + t.reps

    # Формируем массив за последние 7 дней (сегодня и 6 предыдущих)
    result = []
    for i in range(6, -1, -1):
        day = (now - timedelta(days=i)).date().isoformat()
        result.append(DailyReps(day=day, reps=daily.get(day, 0)))
    return result