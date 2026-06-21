FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-libmysqlclient-dev build-essential pkg-config \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove default-libmysqlclient-dev build-essential pkg-config \
    && apt-get install -y --no-install-recommends default-mysql-client \
    && rm -rf /var/lib/apt/lists/*
RUN adduser --disabled-password --gecos "" appuser
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 8001
CMD ["uvicorn", "auth_main:app", "--host", "0.0.0.0", "--port", "8001"]