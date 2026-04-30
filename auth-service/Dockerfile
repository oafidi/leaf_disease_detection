FROM python:3.10-slim

WORKDIR /app

# Dépendances système nécessaires pour le client MySQL
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       default-libmysqlclient-dev \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances en premier pour bénéficier du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ne pas tourner en root en production
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8001

CMD ["uvicorn", "auth_main:app", "--host", "0.0.0.0", "--port", "8001"]