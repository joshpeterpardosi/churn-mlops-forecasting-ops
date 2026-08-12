# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# libgomp1 provides libgomp.so.1, required by LightGBM's native library at
# import time — python:3.13-slim strips it out by default.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000

CMD ["uvicorn", "churn_mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
