from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import Lock

from deep_translator import GoogleTranslator
from fastapi import File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

VERSION = "NAMI_V147B2_AUTOCAP_REAL_WORKER"
MAX_UPLOAD_BYTES = 120 * 1024 * 1024

_PROCESS_LOCK = Lock()
_MODEL_LOCK = Lock()
_WHISPER_MODEL = None


def _load_whisper():
    global _WHISPER_MODEL

    with _MODEL_LOCK:
        if _WHISPER_MODEL is None:
            _WHISPER_MODEL = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1,
            )

    return _WHISPER_MODEL


def _check_owner_key(value: str | None) -> None:
    expected = os.environ.get("NAMI_AUTOCAP_KEY", "").strip()

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="AutoCap owner key is not configured",
        )

    if not value or value != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid AutoCap owner key",
        )


def _run(args: list[str]) -> None:
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])


def _timestamp(seconds: float) -> str:
    total = max(0, round(seconds * 1000))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)

    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{secs:02d},{millis:03d}"
    )


def _translate_batch(texts: list[str]) -> list[str]:
    translator = GoogleTranslator(
        source="zh-CN",
        target="vi",
    )

    try:
        translated = translator.translate_batch(texts)

        if (
            isinstance(translated, list)
            and len(translated) == len(texts)
        ):
            return [
                str(value or "").strip()
                for value in translated
            ]
    except Exception:
        pass

    results: list[str] = []

    for text in texts:
        try:
            value = translator.translate(text)
            results.append(str(value or "").strip())
        except Exception:
            results.append("")

    return results


def _subtitle_filter(path: Path) -> str:
    escaped = str(path).replace("\\", "/")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")

    style = (
        "FontName=Arial,"
        "FontSize=18,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=58"
    )

    return (
        f"subtitles='{escaped}':"
        f"force_style='{style}'"
    )


def register_autocap_routes(app) -> None:
    @app.get("/autocap/health")
    def autocap_health():
        return {
            "ok": True,
            "version": VERSION,
            "worker_ready": True,
            "heavy_work_on_phone": False,
            "source_language": "zh",
            "target_language": "vi",
            "max_upload_mb": 120,
        }

    @app.post("/autocap/process")
    def autocap_process(
        video: UploadFile = File(...),
        x_nami_key: str | None = Header(
            default=None,
            alias="X-NAMI-Key",
        ),
    ):
        _check_owner_key(x_nami_key)

        if not video.filename:
            raise HTTPException(
                status_code=400,
                detail="Missing video filename",
            )

        with _PROCESS_LOCK:
            job_dir = Path(
                tempfile.mkdtemp(prefix="nami_autocap_")
            )

            input_path = job_dir / "input.mp4"
            audio_path = job_dir / "audio.wav"
            zh_path = job_dir / "transcript_zh.txt"
            srt_path = job_dir / "subtitles_vi.srt"
            output_path = job_dir / "vietsub.mp4"

            try:
                total = 0

                with input_path.open("wb") as output:
                    while True:
                        chunk = video.file.read(1024 * 1024)

                        if not chunk:
                            break

                        total += len(chunk)

                        if total > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail="Video exceeds 120 MB",
                            )

                        output.write(chunk)

                if total == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Video is empty",
                    )

                _run([
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

                model = _load_whisper()

                segments, _ = model.transcribe(
                    str(audio_path),
                    language="zh",
                    task="transcribe",
                    vad_filter=True,
                    beam_size=1,
                )

                items = []

                for segment in segments:
                    source = " ".join(
                        segment.text.split()
                    ).strip()

                    if source:
                        items.append({
                            "start": float(segment.start),
                            "end": float(segment.end),
                            "source": source,
                        })

                if not items:
                    raise HTTPException(
                        status_code=422,
                        detail="No Chinese speech detected",
                    )

                translations = _translate_batch([
                    item["source"]
                    for item in items
                ])

                transcript_lines = []
                subtitle_blocks = []

                for index, item in enumerate(items, start=1):
                    vietnamese = translations[index - 1]

                    if not vietnamese:
                        vietnamese = "[Không dịch được câu này]"

                    transcript_lines.append(
                        f'{_timestamp(item["start"])} '
                        f'{item["source"]}'
                    )

                    subtitle_blocks.append(
                        f"{index}\n"
                        f'{_timestamp(item["start"])} --> '
                        f'{_timestamp(item["end"])}\n'
                        f"{vietnamese}\n"
                    )

                zh_path.write_text(
                    "\n".join(transcript_lines) + "\n",
                    encoding="utf-8",
                )

                srt_path.write_text(
                    "\n".join(subtitle_blocks),
                    encoding="utf-8",
                )

                _run([
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vf",
                    _subtitle_filter(srt_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "24",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ])

                if not output_path.is_file():
                    raise RuntimeError(
                        "Output video was not created"
                    )

                return FileResponse(
                    path=str(output_path),
                    media_type="video/mp4",
                    filename="NAMI_AutoCap_vietsub.mp4",
                )

            except HTTPException:
                shutil.rmtree(
                    job_dir,
                    ignore_errors=True,
                )
                raise

            except Exception as exc:
                shutil.rmtree(
                    job_dir,
                    ignore_errors=True,
                )

                raise HTTPException(
                    status_code=500,
                    detail=str(exc),
                ) from exc
