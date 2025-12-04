FROM python:3.10-slim

# Environment: no .pyc files, unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps if needed
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY storage ./storage

RUN mkdir -p storage/chroma

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Run FastAPI app
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]

