
import os
import math
import secrets
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.database import (
    count_active_admins,
    delete_person,
    get_all_users,
    get_attendance,
    get_person_by_id,
    get_registered_faces,
    init_db,
    record_attendance,
    register_person,
    update_user_role,
    update_user_status,
)


MAX_IMAGE_SIZE = 5 * 1024 * 1024
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.363"))
SESSION_DURATION_HOURS = 8

FACE_SERVICE_URL = os.environ["FACE_SERVICE_URL"].rstrip("/")
FACE_SERVICE_TOKEN = os.environ["FACE_SERVICE_TOKEN"]

# In-memory state: suitable for a single backend instance.
retry_attempts = defaultdict(int)
sessions: dict[str, dict] = {}

bearer_scheme = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Face Attendance API",
    version="2.0.0",
    lifespan=lifespan,
)


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
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class RoleUpdate(BaseModel):
    role: Literal["admin", "user"]


class StatusUpdate(BaseModel):
    is_active: bool


@app.get("/")
def root():
    return {
        "service": "Face Attendance API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    # The face model runs in a separate service.
    return {
        "status": "ok",
        "face_service_configured": bool(
            FACE_SERVICE_URL and FACE_SERVICE_TOKEN
        ),
    }


def create_session(person_id: int) -> str:
    """Create an unpredictable, temporary admin session token."""
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "person_id": person_id,
        "expires_at": datetime.now(timezone.utc)
        + timedelta(hours=SESSION_DURATION_HOURS),
    }
    return token


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Admin authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    session = sessions.get(token)

    if not session:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session. Please scan your face again.",
        )

    if datetime.now(timezone.utc) >= session["expires_at"]:
        sessions.pop(token, None)
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please scan your face again.",
        )

    person = get_person_by_id(session["person_id"])

    # Re-check account status and role in PostgreSQL on every request.
    if not person or not person["is_active"]:
        sessions.pop(token, None)
        raise HTTPException(status_code=403, detail="Account is inactive.")

    if person["role"] != "admin":
        sessions.pop(token, None)
        raise HTTPException(status_code=403, detail="Admin access required.")

    return person


async def read_image(file: UploadFile) -> bytes:
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    content = await file.read(MAX_IMAGE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Each image must be 5 MB or smaller.",
        )

    return content


def validate_name(name: str) -> str:
    name = name.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")

    if len(name) > 120:
        raise HTTPException(
            status_code=400,
            detail="Name must be 120 characters or fewer.",
        )

    return name


async def prepare_face_records(
    images: list[UploadFile],
) -> list[tuple[bytes, list[float]]]:
    if not 3 <= len(images) <= 5:
        raise HTTPException(
            status_code=400,
            detail="Upload between 3 and 5 face images.",
        )

    records = []

    for upload in images:
        raw_image = await read_image(upload)
        embedding = await get_face_embedding(raw_image)
        records.append((raw_image, embedding))

    return records


@app.post("/register")
async def register(
    name: str = Form(...),
    images: list[UploadFile] = File(...),
):
    """Public self-registration. New accounts always have the user role."""
    name = validate_name(name)
    records = await prepare_face_records(images)

    person_id = register_person(name, records, "user")

    return {
        "status": "success",
        "message": "Face registered successfully.",
        "name": name,
        "person_id": person_id,
        "role": "user",
    }


def cosine_similarity(
    first: list[float],
    second: list[float],
) -> float:
    if len(first) != len(second) or not first:
        return -1.0

    dot_product = sum(a * b for a, b in zip(first, second))
    norm_first = math.sqrt(sum(a * a for a in first))
    norm_second = math.sqrt(sum(b * b for b in second))

    if norm_first == 0 or norm_second == 0:
        return -1.0

    return dot_product / (norm_first * norm_second)


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
    query_embedding = await get_face_embedding(raw_image)

    best_person = None
    best_score = -1.0

    # Each row: person_id, name, role, stored_embedding.
    # Inactive accounts are excluded by get_registered_faces().
    for person_id, name, role, stored_embedding in get_registered_faces():
        score = cosine_similarity(query_embedding, stored_embedding)

        if score > best_score:
            best_person = (person_id, name, role)
            best_score = score

    if best_person and best_score >= MATCH_THRESHOLD:
        person_id, name, role = best_person

        retry_attempts.pop(session_id, None)
        inserted = record_attendance(person_id)

        response = {
            "status": "success",
            "message": (
                "Attendance recorded."
                if inserted
                else "Attendance already recorded today."
            ),
            "name": name,
            "person_id": person_id,
            "role": role,
            "similarity": round(best_score, 4),
            "attendance_recorded": inserted,
        }

        # Only admins receive an admin session token.
        if role == "admin":
            response["token"] = create_session(person_id)

        return response

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
        "message": "Face not recognized after retry. Please register.",
        "similarity": round(best_score, 4),
    }


# -------------------- ADMIN: ATTENDANCE --------------------


@app.get("/admin/attendance")
def admin_attendance(
    user: str | None = Query(default=None),
    date_filter: date | None = Query(default=None, alias="date"),
    admin: dict = Depends(require_admin),
):
    rows = get_attendance(
        user=user,
        date=date_filter.isoformat() if date_filter else None,
    )

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


# -------------------- ADMIN: USER MANAGEMENT --------------------


@app.get("/admin/users")
def admin_list_users(admin: dict = Depends(require_admin)):
    return {"users": get_all_users()}


@app.post("/admin/users", status_code=201)
async def admin_register_user(
    name: str = Form(...),
    images: list[UploadFile] = File(...),
    role: Literal["admin", "user"] = Form("user"),
    admin: dict = Depends(require_admin),
):
    """Admins can register users and assign an initial role."""
    name = validate_name(name)
    records = await prepare_face_records(images)

    person_id = register_person(name, records, role)
    person = get_person_by_id(person_id)

    return {
        "status": "success",
        "message": "User registered successfully.",
        "user": person,
    }


@app.patch("/admin/users/{person_id}/role")
def admin_update_role(
    person_id: int,
    body: RoleUpdate,
    admin: dict = Depends(require_admin),
):
    target = get_person_by_id(person_id)

    if not target:
        raise HTTPException(status_code=404, detail="User not found.")

    if target["id"] == admin["id"]:
        raise HTTPException(
            status_code=400,
            detail="You cannot change your own role.",
        )

    if (
        target["role"] == "admin"
        and target["is_active"]
        and body.role != "admin"
        and count_active_admins() <= 1
    ):
        raise HTTPException(
            status_code=409,
            detail="Cannot demote the last active admin.",
        )

    if not update_user_role(person_id, body.role):
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "status": "success",
        "message": "User role updated.",
        "user": get_person_by_id(person_id),
    }


@app.patch("/admin/users/{person_id}/status")
def admin_update_status(
    person_id: int,
    body: StatusUpdate,
    admin: dict = Depends(require_admin),
):
    target = get_person_by_id(person_id)

    if not target:
        raise HTTPException(status_code=404, detail="User not found.")

    if target["id"] == admin["id"] and not body.is_active:
        raise HTTPException(
            status_code=400,
            detail="You cannot deactivate your own account.",
        )

    if (
        target["role"] == "admin"
        and target["is_active"]
        and not body.is_active
        and count_active_admins() <= 1
    ):
        raise HTTPException(
            status_code=409,
            detail="Cannot deactivate the last active admin.",
        )

    if not update_user_status(person_id, body.is_active):
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "status": "success",
        "message": "User status updated.",
        "user": get_person_by_id(person_id),
    }


@app.delete("/admin/users/{person_id}")
def admin_delete_user(
    person_id: int,
    admin: dict = Depends(require_admin),
):
    target = get_person_by_id(person_id)

    if not target:
        raise HTTPException(status_code=404, detail="User not found.")

    if target["id"] == admin["id"]:
        raise HTTPException(
            status_code=400,
            detail="You cannot delete your own account.",
        )

    if (
        target["role"] == "admin"
        and target["is_active"]
        and count_active_admins() <= 1
    ):
        raise HTTPException(
            status_code=409,
            detail="Cannot delete the last active admin.",
        )

    delete_person(person_id)

    # Invalidate any in-memory sessions belonging to this person.
    for token, session in list(sessions.items()):
        if session["person_id"] == person_id:
            sessions.pop(token, None)

    return {
        "status": "success",
        "message": "User permanently deleted.",
        "person_id": person_id,
    }


# -------------------- FACE MODEL SERVICE --------------------


async def get_face_embedding(image_bytes: bytes) -> list[float]:
    headers = {
        "Authorization": f"Bearer {FACE_SERVICE_TOKEN}"
    }
    timeout = httpx.Timeout(15.0, connect=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{FACE_SERVICE_URL}/embedding",
                headers=headers,
                files={
                    "image": (
                        "face.jpg",
                        image_bytes,
                        "image/jpeg",
                    )
                },
            )

        if response.status_code == 400:
            try:
                detail = response.json().get(
                    "detail", "Invalid face image."
                )
            except ValueError:
                detail = "Invalid face image."

            raise HTTPException(status_code=400, detail=detail)

        response.raise_for_status()
        result = response.json()

        if result.get("status") != "success":
            raise HTTPException(
                status_code=502,
                detail="Face recognition service returned an invalid response.",
            )

        embedding = result.get("embedding")

        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in embedding
            )
        ):
            raise HTTPException(
                status_code=502,
                detail="Face recognition service returned an invalid embedding.",
            )

        return embedding

    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service is unavailable.",
        ) from exc
