FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ne pas tourner en root en production
RUN adduser --disabled-password --gecos "" appuser
USER appuser

CMD ["uvicorn", "vision_main:app", "--host", "0.0.0.0", "--port", "8000"]