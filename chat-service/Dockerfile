FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN adduser --disabled-password --gecos "" appuser
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 8003
CMD ["uvicorn", "chat_main:app", "--host", "0.0.0.0", "--port", "8003"]