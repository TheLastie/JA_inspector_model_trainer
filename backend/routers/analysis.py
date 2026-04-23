from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import uuid
import os
import cv2
import numpy as np
from datetime import datetime

from auth import get_current_user, get_db
from models import User, Training
from schemas import AnalysisResponse
from database import SessionLocal
from data_pipeline import load_all_frames
from joint_analysis_video import (
    process_video_with_yolo, segment_video_simple,
    compute_joint_errors, compute_exercise_score, draw_skeleton_with_errors,
    find_reference_json, convert_score_to_5star
)
static_results_dir = "static/results"
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# Глобальные переменные будут установлены из main.py при старте
ensemble_models = {}
classifier = None
scaler = None
label_encoder = None
yolo_model = None
device = None
thresholds_cache = {}

def get_thresholds(exercise: str) -> np.ndarray:
    if exercise in thresholds_cache:
        return thresholds_cache[exercise]
    ref_json = find_reference_json(exercise)
    if ref_json and os.path.exists(ref_json):
        try:
            angles, _ = load_all_frames(ref_json)
            if len(angles) > 0:
                model = ensemble_models.get(exercise)
                if model:
                    ref_norm = (angles / np.pi) * 2.0 - 1.0
                    ref_errors = compute_joint_errors(model, ref_norm, device, seq_len=30)
                    thresh = np.percentile(ref_errors, 95.0, axis=0)
                    thresholds_cache[exercise] = thresh
                    return thresh
        except Exception:
            pass
    return np.full(11, 0.1, dtype=np.float32)

def analyze_video_task(video_path: str, output_dir: str, manual_exercise: Optional[str], user_id: str):
    frames, all_angles, all_keypoints, all_boxes = process_video_with_yolo(
        video_path, "yolo26x-pose.pt", device, quiet=True
    )
    valid_indices = [i for i, a in enumerate(all_angles) if a is not None]
    if not valid_indices:
        raise ValueError("No pose detected in video")

    if manual_exercise and manual_exercise in ensemble_models:
        segments = [(valid_indices[0], valid_indices[-1], manual_exercise, 1.0)]
    else:
        segments = segment_video_simple(
            all_angles, valid_indices, ensemble_models, classifier, scaler, label_encoder,
            device, seq_len=30, window_stride=15, smooth_window=15, min_segment_frames=30
        )

    all_errors = np.full((len(frames), 11), np.nan)
    all_thresholds = np.full((len(frames), 11), np.nan)
    segment_info = []

    for seg in segments:
        start_frame, end_frame, ex_name, conf = seg
        seg_indices = [i for i in range(start_frame, end_frame+1) if i in valid_indices]
        if not seg_indices:
            continue
        model = ensemble_models.get(ex_name)
        score_10 = None
        score_5 = None
        if model:
            angles_seg = np.stack([all_angles[i] for i in seg_indices])
            angles_norm = (angles_seg / np.pi) * 2.0 - 1.0
            thresholds = get_thresholds(ex_name)
            errors = compute_joint_errors(model, angles_norm, device, seq_len=30)
            score_10 = compute_exercise_score(errors, thresholds)
            score_5 = convert_score_to_5star(score_10)
            for i, idx in enumerate(seg_indices):
                all_errors[idx] = errors[i]
                all_thresholds[idx] = thresholds

        segment_info.append({
            'start_frame': start_frame, 'end_frame': end_frame,
            'exercise': ex_name, 'confidence': float(conf),
            'score_10': score_10, 'score_5': score_5
        })

    annotated_frames = []
    for i, frame in enumerate(frames):
        errs = all_errors[i]
        kpts = all_keypoints[i]
        thresh = all_thresholds[i] if not np.isnan(all_thresholds[i]).all() else None
        current_ex = "none"
        model_avail = False
        for seg in segment_info:
            if seg['start_frame'] <= i <= seg['end_frame']:
                current_ex = seg['exercise']
                model_avail = seg['score_10'] is not None
                break
        if np.isnan(errs).all():
            errs = np.zeros(11)
            model_avail = False
        annotated = draw_skeleton_with_errors(frame.copy(), kpts, errs, thresh, model_avail)
        cv2.putText(annotated, f"Ex: {current_ex}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        if model_avail:
            active_seg = next(s for s in segment_info if s['start_frame'] <= i <= s['end_frame'])
            if active_seg['score_10'] is not None:
                cv2.putText(annotated, f"Score: {active_seg['score_10']:.1f}/10 ({active_seg['score_5']}*)",
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)
        annotated_frames.append(annotated)

    os.makedirs(output_dir, exist_ok=True)
    video_name = f"annotated_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(output_dir, video_name)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30, (w, h))
    for f in annotated_frames:
        out.write(f)
    out.release()

    overall_score = np.mean([s['score_10'] for s in segment_info if s['score_10'] is not None]) if segment_info else None
    overall_score_5 = convert_score_to_5star(overall_score) if overall_score else None

    db = SessionLocal()
    try:
        training = Training(
            user_id=user_id,
            exercise_name=manual_exercise or ", ".join(set(s['exercise'] for s in segment_info)),
            date=datetime.utcnow().isoformat(),
            reps=0,
            errors=[],
            duration=0,
            score_10=overall_score,
            score_5=overall_score_5,
            segments=segment_info,
            video_url=f"/static/results/{video_name}"
        )
        db.add(training)
        db.commit()
        training_id = training.id
    finally:
        db.close()

    return {
        "video_url": f"/static/results/{video_name}",
        "segments": segment_info,
        "overall_score": overall_score,
        "overall_score_5": overall_score_5,
        "training_id": training_id
    }

@router.post("/upload", response_model=AnalysisResponse)
async def upload_video(
    file: UploadFile = File(...),
    exercise: Optional[str] = Form(None),
    user: User = Depends(get_current_user)
):
    try:
        temp_dir = "static/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_id = uuid.uuid4().hex
        input_path = os.path.join(temp_dir, f"{temp_id}.mp4")
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        if not ensemble_models:
            raise HTTPException(status_code=503, detail="Models not loaded")

        result = analyze_video_task(
    video_path=input_path,
    output_dir=static_results_dir,
    manual_exercise=exercise,
    user_id=user.id
)
        os.unlink(input_path)
        return AnalysisResponse(**result)
    except Exception as e:
        if os.path.exists(input_path):
            os.unlink(input_path)
        return AnalysisResponse(video_url="", segments=[], overall_score=None, overall_score_5=None, training_id=None, error=str(e))