FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# Optional: pre-download the embedding model at build time when the
# sentence-transformers dependency is installed.
RUN python - <<'PY'
try:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer('all-MiniLM-L6-v2')
    print('Embedding model cached.')
except Exception as e:
    print('Skipping embedding model cache:', e)
PY

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
