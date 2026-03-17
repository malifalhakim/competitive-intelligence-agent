FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY document_converter.py .
COPY create_chunks.py .
COPY vector_store.py .
COPY agent.py .
COPY init_db.py .
COPY api.py .