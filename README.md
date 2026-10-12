# Face Attendance Backend API

A FastAPI backend for a face-recognition attendance application.
The frontend is maintained and deployed separately.

## Project structure

- `app/main.py`: API endpoints
- `app/database.py`: PostgreSQL operations
- `app/schemas.py`: API schemas
- `model/face_model.py`: Face detection and recognition

## Requirements

- Python 3.11+
- PostgreSQL
- YuNet face detection ONNX model
- SFace face recognition ONNX model

## Model files

Download the two ONNX models from OpenCV Zoo and put them
directly inside the `model/` directory:

- `model/face_detection_yunet.onnx`
- `model/face_recognition_sface.onnx`

OpenCV Zoo:
https://github.com/opencv/opencv_zoo

The model binaries are not included in this repository.

## Environment variables

- `DATABASE_URL`: PostgreSQL connection URL
- `FRONTEND_ORIGINS`: Comma-separated frontend origins
- `MATCH_THRESHOLD`: Optional face similarity threshold;
  defaults to `0.363`

Example local frontend origins:

`http://localhost:3000,http://127.0.0.1:5500`

Replace these with your actual deployed frontend origin(s)
when deploying.

## Run locally

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

Set `DATABASE_URL` and `FRONTEND_ORIGINS`, then start the API:

    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API documentation:

    http://localhost:8000/docs

Health check:

    http://localhost:8000/health

## Endpoints

- `GET /health`
- `POST /register`
- `POST /recognize`
- `GET /admin/attendance`

## Deploy to Render

1. Create a PostgreSQL database.
2. Push this repository to GitHub.
3. Create a Render Docker Web Service using the repository.
4. Set `DATABASE_URL` to the PostgreSQL connection URL.
5. Set `FRONTEND_ORIGINS` to the deployed frontend URL.
6. Ensure both ONNX model files are present in `model/`.
7. Deploy and test `/health` and `/docs`.

## Security and limitations

- Add authentication to `/admin/attendance` before exposing it publicly.
- Obtain consent before collecting biometric data.
- Restrict access to stored face images and embeddings.
- Use HTTPS in production.
- This prototype does not include liveness detection.
- The recognition threshold must be calibrated with representative
  images before relying on attendance results.
- The retry counter is held in memory and assumes one backend instance.
