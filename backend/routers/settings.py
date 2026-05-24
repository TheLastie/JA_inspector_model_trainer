from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas import SettingsUpdate, SettingsOut
from models import UserSettings
from auth import get_current_user, get_db

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/", response_model=SettingsOut)
def get_settings(user=Depends(get_current_user), db: Session = Depends(get_db)):
    settings = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()
    if not settings:
        settings = UserSettings(user_id=user.id)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return SettingsOut(
        strictness=settings.strictness,
        hintsEnabled=settings.hints_enabled,
        darkMode=settings.dark_mode,
        reminderTime=settings.reminder_time,
        reminderDays=settings.reminder_days
    )

@router.put("/", response_model=dict)
def update_settings(
    data: SettingsUpdate,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    settings = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()
    if not settings:
        settings = UserSettings(user_id=user.id)
        db.add(settings)

    for field, value in data.dict(exclude_unset=True).items():
        if field == "hintsEnabled":
            settings.hints_enabled = value
        elif field == "darkMode":
            settings.dark_mode = value
        elif field == "reminderTime":
            settings.reminder_time = value
        elif field == "reminderDays":
            settings.reminder_days = value
        elif field == "strictness":
            settings.strictness = value

    db.commit()
    return {"success": True}