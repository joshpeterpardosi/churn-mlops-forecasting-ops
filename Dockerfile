# Dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000

CMD ["uvicorn", "churn_mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
