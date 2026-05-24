from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional

# ── Auth ──
class UserRegister(BaseModel):
    email: EmailStr
    password: str

    @validator('password')
    def password_min_length(cls, v):
        if len(v) < 6:
            raise ValueError('Пароль должен содержать минимум 6 символов')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    token: str
    user: dict  # {id, email}

# ── Trainings ──
class TrainingCreate(BaseModel):
    exerciseName: str
    date: str
    reps: int
    errors: List[str] = []
    duration: int = 0
    score_10: Optional[float] = None
    score_5: Optional[int] = None
    joint_errors: Optional[List[dict]] = None
    segments: Optional[List[dict]] = None
    video_url: Optional[str] = None

class TrainingOut(TrainingCreate):
    id: str

    class Config:
        from_attributes = True   

# ── Settings ──
class SettingsUpdate(BaseModel):
    strictness: Optional[str] = None
    hintsEnabled: Optional[bool] = None
    darkMode: Optional[bool] = None
    reminderTime: Optional[str] = None
    reminderDays: Optional[List[int]] = None

class SettingsOut(BaseModel):
    strictness: str
    hintsEnabled: bool
    darkMode: bool
    reminderTime: str
    reminderDays: List[int]

    class Config:
        from_attributes = True

# ── Stats ──
class MuscleStatsOut(BaseModel):
    legs: int
    core: int
    arms: int
    total: int

class DailyReps(BaseModel):
    day: str
    reps: int

# ── Analysis ──
class AnalysisResponse(BaseModel):
    video_url: str
    segments: List[dict]
    overall_score: Optional[float]
    overall_score_5: Optional[int]
    training_id: Optional[str]
    error: Optional[str] = None