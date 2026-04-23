from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from schemas import UserRegister, UserLogin, TokenResponse
from models import User, UserSettings, MuscleStats
from auth import hash_password, verify_password, create_access_token, get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/register", response_model=TokenResponse, status_code=201)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")
    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password)
    )
    db.add(user)
    db.flush()
    db.add(UserSettings(user_id=user.id))
    db.add(MuscleStats(user_id=user.id))
    db.commit()
    token = create_access_token(user.id)
    return TokenResponse(token=token, user={"id": user.id, "email": user.email})

@router.post("/login", response_model=TokenResponse)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.id)
    return TokenResponse(token=token, user={"id": user.id, "email": user.email})