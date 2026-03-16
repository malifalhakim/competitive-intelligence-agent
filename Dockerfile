FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY document_converter.py .

ENTRYPOINT ["python", "document_converter.py"]