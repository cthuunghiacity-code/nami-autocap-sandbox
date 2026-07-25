from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import Lock

import edge_tts
from deep_translator import GoogleTranslator
from fastapi import File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

VERSION = "NAMI_V147D_SUBTITLE_AND_VIETNAMESE_DUB"
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
        raise RuntimeError(result.stderr[-5000:])


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


async def _create_voice(
    text: str,
    output_path: Path,
    voice: str,
) -> None:
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+8%",
        volume="+0%",
        pitch="+0Hz",
    )

    await communicator.save(str(output_path))


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


def _video_has_audio(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    return bool(result.stdout.strip())


def _render_subtitle_only(
    input_path: Path,
    srt_path: Path,
    output_path: Path,
) -> None:
    args = [
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
    ]

    if _video_has_audio(input_path):
        args.extend([
            "-c:a",
            "aac",
            "-b:a",
            "128k",
        ])
    else:
        args.append("-an")

    args.extend([
        "-movflags",
        "+faststart",
        str(output_path),
    ])

    _run(args)


def _render_with_dubbing(
    input_path: Path,
    srt_path: Path,
    voice_files: list[tuple[Path, int]],
    output_path: Path,
) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
    ]

    for voice_path, _ in voice_files:
        args.extend(["-i", str(voice_path)])

    filters: list[str] = []
    mix_labels: list[str] = []

    if _video_has_audio(input_path):
        filters.append("[0:a]volume=0.16[original]")
        mix_labels.append("[original]")
    else:
        filters.append(
            "anullsrc=channel_layout=stereo:"
            "sample_rate=44100[original]"
        )
        mix_labels.append("[original]")

    for index, (_, delay_ms) in enumerate(
        voice_files,
        start=1,
    ):
        label = f"voice{index}"

        filters.append(
            f"[{index}:a]"
            f"aresample=44100,"
            f"adelay={delay_ms}|{delay_ms},"
            f"volume=1.35"
            f"[{label}]"
        )

        mix_labels.append(f"[{label}]")

    mix_inputs = "".join(mix_labels)

    filters.append(
        f"{mix_inputs}"
        f"amix=inputs={len(mix_labels)}:"
        f"duration=first:"
        f"dropout_transition=0:"
        f"normalize=0"
        f"[mixed]"
    )

    _run([
        *args,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "0:v:0",
        "-map",
        "[mixed]",
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
        "160k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ])


def register_autocap_routes(app) -> None:
    @app.get("/autocap/health")
    def autocap_health():
        return {
            "ok": True,
            "version": VERSION,
            "worker_ready": True,
            "heavy_work_on_phone": False,
            "subtitle_mode": True,
            "vietnamese_dubbing_mode": True,
            "voices": {
                "female": "vi-VN-HoaiMyNeural",
                "male": "vi-VN-NamMinhNeural",
            },
            "max_upload_mb": 120,
        }

    @app.post("/autocap/process")
    def autocap_process(
        video: UploadFile = File(...),
        mode: str = Form(default="subtitle"),
        voice: str = Form(
            default="vi-VN-HoaiMyNeural"
        ),
        x_nami_key: str | None = Header(
            default=None,
            alias="X-NAMI-Key",
        ),
    ):
        _check_owner_key(x_nami_key)

        if mode not in {"subtitle", "dub"}:
            raise HTTPException(
                status_code=400,
                detail="Unsupported processing mode",
            )

        allowed_voices = {
            "vi-VN-HoaiMyNeural",
            "vi-VN-NamMinhNeural",
        }

        if voice not in allowed_voices:
            voice = "vi-VN-HoaiMyNeural"

        if not video.filename:
            raise HTTPException(
                status_code=400,
                detail="Missing video filename",
            )

        with _PROCESS_LOCK:
            job_dir = Path(
                tempfile.mkdtemp(
                    prefix="nami_autocap_"
                )
            )

            input_path = job_dir / "input.mp4"
            audio_path = job_dir / "audio.wav"
            srt_path = job_dir / "subtitles_vi.srt"
            output_path = job_dir / "result.mp4"

            try:
                total = 0

                with input_path.open("wb") as output:
                    while True:
                        chunk = video.file.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        total += len(chunk)

                        if total > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    "Video exceeds 120 MB"
                                ),
                            )

                        output.write(chunk)

                if total == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Video is empty",
                    )

                if not _video_has_audio(input_path):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Video không có âm thanh "
                            "để nhận dạng lời nói."
                        ),
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
                            "start": float(
                                segment.start
                            ),
                            "end": float(
                                segment.end
                            ),
                            "source": source,
                        })

                if not items:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Không nhận dạng được "
                            "lời nói tiếng Trung."
                        ),
                    )

                translations = _translate_batch([
                    item["source"]
                    for item in items
                ])

                subtitle_blocks = []

                for index, item in enumerate(
                    items,
                    start=1,
                ):
                    vietnamese = translations[
                        index - 1
                    ]

                    if not vietnamese:
                        vietnamese = (
                            "[Không dịch được câu này]"
                        )

                    item["vietnamese"] = vietnamese

                    subtitle_blocks.append(
                        f"{index}\n"
                        f'{_timestamp(item["start"])} '
                        f'--> {_timestamp(item["end"])}\n'
                        f"{vietnamese}\n"
                    )

                srt_path.write_text(
                    "\n".join(subtitle_blocks),
                    encoding="utf-8",
                )

                if mode == "dub":
                    voice_files = []

                    for index, item in enumerate(
                        items,
                        start=1,
                    ):
                        voice_path = (
                            job_dir
                            / f"voice_{index:04d}.mp3"
                        )

                        asyncio.run(
                            _create_voice(
                                str(
                                    item["vietnamese"]
                                ),
                                voice_path,
                                voice,
                            )
                        )

                        voice_files.append((
                            voice_path,
                            round(
                                float(item["start"])
                                * 1000
                            ),
                        ))

                    _render_with_dubbing(
                        input_path,
                        srt_path,
                        voice_files,
                        output_path,
                    )
                else:
                    _render_subtitle_only(
                        input_path,
                        srt_path,
                        output_path,
                    )

                if not output_path.is_file():
                    raise RuntimeError(
                        "Output video was not created"
                    )

                filename = (
                    "NAMI_AutoCap_long_tieng_Viet.mp4"
                    if mode == "dub"
                    else "NAMI_AutoCap_vietsub.mp4"
                )

                return FileResponse(
                    path=str(output_path),
                    media_type="video/mp4",
                    filename=filename,
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
