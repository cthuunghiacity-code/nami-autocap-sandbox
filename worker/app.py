from __future__ import annotations

import html
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import Lock
from typing import Final

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from transformers import pipeline

APP_VERSION: Final[str] = "NAMI_V147B1"
MAX_UPLOAD_BYTES: Final[int] = 150 * 1024 * 1024

app = FastAPI(
    title="NAMI AutoCap Worker",
    version="0.1.0",
)

model_lock = Lock()
processing_lock = Lock()

whisper_model: WhisperModel | None = None
translator = None


def load_models() -> None:
    global whisper_model, translator

    with model_lock:
        if whisper_model is None:
            whisper_model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1,
            )

        if translator is None:
            translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-zh-vi",
                device=-1,
            )


def run_command(args: list[str]) -> None:
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        error = completed.stderr[-3000:]
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {error}"
        )


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)

    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{secs:02d},{millis:03d}"
    )


def translate_text(text: str) -> str:
    cleaned = " ".join(text.split()).strip()

    if not cleaned:
        return ""

    if translator is None:
        raise RuntimeError("Translator is unavailable")

    result = translator(
        cleaned,
        max_length=256,
        truncation=True,
    )

    translated = result[0]["translation_text"]
    return " ".join(translated.split()).strip()


def escape_subtitle_path(path: Path) -> str:
    value = str(path).replace("\\", "/")
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "NAMI AutoCap Worker",
        "version": APP_VERSION,
        "ready": True,
        "heavy_work_on_phone": False,
        "source_language": "zh",
        "target_language": "vi",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "worker_ready": True,
    }


@app.post("/process")
def process_video(
    video: UploadFile = File(...),
) -> FileResponse:
    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail="Missing filename",
        )

    with processing_lock:
        job_dir = Path(
            tempfile.mkdtemp(
                prefix="nami_",
                dir="/tmp/nami_jobs",
            )
        )

        input_path = job_dir / "input.mp4"
        audio_path = job_dir / "audio.wav"
        srt_path = job_dir / "subtitles_vi.srt"
        output_path = job_dir / "vietsub.mp4"

        try:
            total = 0

            with input_path.open("wb") as handle:
                while True:
                    chunk = video.file.read(1024 * 1024)

                    if not chunk:
                        break

                    total += len(chunk)

                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Video exceeds 150 MB",
                        )

                    handle.write(chunk)

            if total == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Empty video",
                )

            run_command([
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_path),
            ])

            load_models()

            if whisper_model is None:
                raise RuntimeError("Whisper is unavailable")

            segments, info = whisper_model.transcribe(
                str(audio_path),
                language="zh",
                task="transcribe",
                vad_filter=True,
                beam_size=1,
            )

            subtitle_entries: list[str] = []
            index = 1

            for segment in segments:
                source = segment.text.strip()

                if not source:
                    continue

                vietnamese = translate_text(source)

                if not vietnamese:
                    continue

                subtitle_entries.append(
                    f"{index}\n"
                    f"{srt_timestamp(segment.start)} --> "
                    f"{srt_timestamp(segment.end)}\n"
                    f"{vietnamese}\n"
                )
                index += 1

            if not subtitle_entries:
                raise HTTPException(
                    status_code=422,
                    detail="No Chinese speech detected",
                )

            srt_path.write_text(
                "\n".join(subtitle_entries),
                encoding="utf-8",
            )

            subtitle_filter = (
                "subtitles='"
                + escape_subtitle_path(srt_path)
                + "':force_style='"
                + "FontName=Arial,"
                + "FontSize=18,"
                + "PrimaryColour=&H00FFFFFF,"
                + "OutlineColour=&H00000000,"
                + "BorderStyle=1,"
                + "Outline=2,"
                + "Shadow=0,"
                + "Alignment=2,"
                + "MarginV=55"
                + "'"
            )

            run_command([
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                subtitle_filter,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output_path),
            ])

            if not output_path.is_file():
                raise RuntimeError("Output video was not created")

            return FileResponse(
                path=output_path,
                media_type="video/mp4",
                filename="NAMI_AutoCap_vietsub.mp4",
                background=None,
            )

        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc
