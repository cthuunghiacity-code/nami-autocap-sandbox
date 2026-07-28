FROM python:3.11-slim AS model-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HOME=/tmp/huggingface

RUN pip install --no-cache-dir \
    ctranslate2==4.8.1 \
    transformers==4.53.2 \
    sentencepiece==0.2.0 \
    sacremoses==0.1.1 \
    torch==2.7.1

RUN mkdir -p \
    /opt/nami_models/opus-mt-zh-vi-ct2 \
    /opt/nami_models/opus-mt-zh-vi-tokenizer

RUN ct2-transformers-converter \
    --model Helsinki-NLP/opus-mt-zh-vi \
    --output_dir /opt/nami_models/opus-mt-zh-vi-ct2 \
    --quantization int8

RUN python - <<'PY'
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-zh-vi"
)

tokenizer.save_pretrained(
    "/opt/nami_models/opus-mt-zh-vi-tokenizer"
)
PY


FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/.huggingface
ENV OMP_NUM_THREADS=2
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libgomp1 \
      tesseract-ocr \
      tesseract-ocr-chi-sim \
      fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r /app/requirements.txt

COPY --from=model-builder \
    /opt/nami_models \
    /opt/nami_models

COPY app.py /app/app.py
COPY autocap_routes.py /app/autocap_routes.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
