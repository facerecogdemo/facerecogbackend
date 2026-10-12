FROM python:3.11-slim

WORKDIR /service

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
