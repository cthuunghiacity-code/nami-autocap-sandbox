FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/.huggingface
ENV TRANSFORMERS_CACHE=/data/.huggingface
ENV OMP_NUM_THREADS=2

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r /app/requirements.txt

COPY app.py /app/app.py

RUN mkdir -p /data/.huggingface /tmp/nami_jobs

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
