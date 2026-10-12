import os
from collections import defaultdict
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.database import (
    get_attendance,
    get_registered_faces,
    init_db,
    record_attendance,
    register_person,
)
from model.face_model import FaceModel


MAX_IMAGE_SIZE = 5 * 1024 * 1024

# This is a starting value. Calibrate it with real images.
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.363"))

# This in-memory retry counter assumes a single backend instance.
retry_attempts = defaultdict(int)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.face_model = FaceModel()
    yield


app = FastAPI(
    title="Face Attendance API",
    version="1.0.0",
    lifespan=lifespan,
)


# Set FRONTEND_ORIGINS to your frontend's exact origin.
# Multiple origins can be comma-separated.
frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
def root():
    return {
        "service": "Face Attendance API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "face_model"),
    }


async def read_image(file: UploadFile):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    content = await file.read(MAX_IMAGE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Each image must be 5 MB or smaller.",
        )

    return content


@app.post("/register")
async def register(
    name: str = Form(...),
    images: list[UploadFile] = File(...),
):
    name = name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name is required.",
        )

    if len(name) > 120:
        raise HTTPException(
            status_code=400,
            detail="Name must be 120 characters or fewer.",
        )

    if not 3 <= len(images) <= 5:
        raise HTTPException(
            status_code=400,
            detail="Upload between 3 and 5 face images.",
        )

    records = []

    for upload in images:
        raw_image = await read_image(upload)

        try:
            embedding = app.state.face_model.get_embedding(raw_image)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        records.append((raw_image, embedding))

    person_id = register_person(name, records)

    return {
        "status": "success",
        "message": "Face registered successfully.",
        "name": name,
        "person_id": person_id,
    }


@app.post("/recognize")
async def recognize(
    image: UploadFile = File(...),
    session_id: str = Form(...),
):
    try:
        session_id = str(UUID(session_id))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="session_id must be a valid UUID.",
        )

    raw_image = await read_image(image)

    try:
        query_embedding = app.state.face_model.get_embedding(raw_image)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    best_person = None
    best_score = -1.0

    registered_faces = get_registered_faces()

    for person_id, name, stored_embedding in registered_faces:
        score = app.state.face_model.similarity(
            query_embedding,
            stored_embedding,
        )

        if score > best_score:
            best_person = (person_id, name)
            best_score = score

    if best_person and best_score >= MATCH_THRESHOLD:
        person_id, name = best_person

        retry_attempts.pop(session_id, None)

        inserted = record_attendance(person_id)

        return {
            "status": "success",
            "message": (
                "Attendance recorded."
                if inserted
                else "Attendance already recorded today."
            ),
            "name": name,
            "person_id": person_id,
            "similarity": round(best_score, 4),
            "attendance_recorded": inserted,
        }

    retry_attempts[session_id] += 1
    attempts = retry_attempts[session_id]

    if attempts == 1:
        return {
            "status": "retry",
            "message": "Face not recognized. Please try once more.",
            "similarity": round(best_score, 4),
        }

    retry_attempts.pop(session_id, None)

    return {
        "status": "register_required",
        "message": (
            "Face not recognized after retry. Please register."
        ),
        "similarity": round(best_score, 4),
    }


@app.get("/admin/attendance")
def admin_attendance(
    user: str | None = Query(default=None),
    date: str | None = Query(default=None),
):
    # TODO: Implement admin authentication before public deployment.
    rows = get_attendance(user=user, date=date)

    return {
        "records": [
            {
                "id": person_id,
                "name": name,
                "date": day.isoformat(),
                "timestamp": timestamp.isoformat(),
            }
            for person_id, name, day, timestamp in rows
        ]
    }
