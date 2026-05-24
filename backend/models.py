from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from database import Base
import uuid

def gen_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(String, default="datetime('now')")

    trainings = relationship("Training", back_populates="user", cascade="all, delete")
    settings = relationship("UserSettings", uselist=False, back_populates="user", cascade="all, delete")
    muscle_stats = relationship("MuscleStats", uselist=False, back_populates="user", cascade="all, delete")

class Training(Base):
    __tablename__ = "trainings"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    exercise_name = Column(String, nullable=False)
    date = Column(String, nullable=False)
    reps = Column(Integer, nullable=False)
    errors = Column(JSON, nullable=False)
    duration = Column(Integer, default=0)

    # Поля для результатов анализа видео (могут быть NULL)
    score_10 = Column(Float, nullable=True)          # итоговая оценка 0–10
    score_5 = Column(Integer, nullable=True)         # оценка 1–5 звёзд
    joint_errors = Column(JSON, nullable=True)       # детальные ошибки по суставам
    segments = Column(JSON, nullable=True)           # сегменты упражнений
    video_url = Column(String, nullable=True)        # путь к аннотированному видео

    user = relationship("User", back_populates="trainings")

class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    strictness = Column(String, default="medium")
    hints_enabled = Column(Boolean, default=True)
    dark_mode = Column(Boolean, default=False)
    reminder_time = Column(String, default="09:00")
    reminder_days = Column(JSON, default=[1,2,3,4,5])

    user = relationship("User", back_populates="settings")

class MuscleStats(Base):
    __tablename__ = "muscle_stats"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    legs = Column(Integer, default=0)
    core = Column(Integer, default=0)
    arms = Column(Integer, default=0)
    total = Column(Integer, default=0)

    user = relationship("User", back_populates="muscle_stats")