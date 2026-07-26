from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import cv2
import pytesseract

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None
from difflib import SequenceMatcher
from pytesseract import Output
from pathlib import Path
from threading import Lock

import edge_tts
from deep_translator import GoogleTranslator
from fastapi import File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

VERSION = "NAMI_V147O_RAPIDOCR_CHINESE_CAPTIONS"
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



def _extract_audio_for_sync(
    input_path: Path,
    audio_path: Path,
) -> None:
    if audio_path.is_file() and audio_path.stat().st_size > 0:
        return

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


def _detect_chinese_speech_windows(
    audio_path: Path,
) -> list[dict]:
    model = _load_whisper()

    segments, _ = model.transcribe(
        str(audio_path),
        language="zh",
        task="transcribe",
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
    )

    windows: list[dict] = []

    for segment in segments:
        start = max(0.0, float(segment.start))
        end = max(start + 0.12, float(segment.end))
        spoken_text = " ".join(
            str(segment.text or "").split()
        ).strip()

        windows.append({
            "start": start,
            "end": end,
            "text": spoken_text,
        })

    return windows


def _time_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(
        0.0,
        min(end_a, end_b) - max(start_a, start_b),
    )



def _han_only(value: str) -> str:
    return "".join(
        character
        for character in str(value or "")
        if (
            "\u3400" <= character <= "\u4dbf"
            or "\u4e00" <= character <= "\u9fff"
        )
    )


def _chinese_source_similarity(
    ocr_text: str,
    speech_text: str,
) -> float:
    ocr_han = _han_only(ocr_text)
    speech_han = _han_only(speech_text)

    if not ocr_han or not speech_han:
        return 0.0

    sequence_score = SequenceMatcher(
        None,
        ocr_han,
        speech_han,
    ).ratio()

    ocr_chars = set(ocr_han)
    speech_chars = set(speech_han)
    shared = len(ocr_chars & speech_chars)

    coverage_score = shared / max(
        1,
        min(len(ocr_chars), len(speech_chars)),
    )

    length_score = min(
        len(ocr_han),
        len(speech_han),
    ) / max(
        len(ocr_han),
        len(speech_han),
    )

    return (
        sequence_score * 0.60
        + coverage_score * 0.30
        + length_score * 0.10
    )


def _select_chinese_translation_source(
    item: dict,
) -> dict:
    updated = dict(item)

    ocr_text = " ".join(
        str(item.get("source") or "").split()
    ).strip()

    speech_text = " ".join(
        str(
            item.get("recognized_speech") or ""
        ).split()
    ).strip()

    ocr_han = _han_only(ocr_text)
    speech_han = _han_only(speech_text)

    updated["ocr_source"] = ocr_text
    updated["whisper_source"] = speech_text

    if not speech_han:
        updated["translation_source"] = ocr_text
        updated["source_selection"] = (
            "ocr_no_valid_whisper"
        )
        updated["source_similarity"] = 1.0
        return updated

    if not ocr_han:
        updated["translation_source"] = speech_text
        updated["source_selection"] = (
            "whisper_invalid_ocr"
        )
        updated["source_similarity"] = 0.0
        return updated

    similarity = _chinese_source_similarity(
        ocr_text,
        speech_text,
    )

    updated["source_similarity"] = round(
        similarity,
        4,
    )

    short_pair = (
        len(ocr_han) <= 4
        and len(speech_han) <= 4
    )

    has_shared_character = bool(
        set(ocr_han) & set(speech_han)
    )

    # OCR thường giữ đúng chữ tên riêng và dấu câu hơn
    # khi hai nguồn thật sự khớp.
    use_ocr = (
        similarity >= 0.48
        or (
            short_pair
            and has_shared_character
            and similarity >= 0.32
        )
    )

    if use_ocr:
        updated["translation_source"] = ocr_text
        updated["source_selection"] = (
            "ocr_confirmed_by_whisper"
        )
    else:
        # Hai nguồn lệch mạnh: không được đem OCR sai
        # như “八个国家”, “综合”, “和二加一” đi dịch.
        updated["translation_source"] = speech_text
        updated["source_selection"] = (
            "whisper_replaced_mismatched_ocr"
        )

    return updated


def _cross_check_translation_sources(
    items: list[dict],
) -> list[dict]:
    return [
        _select_chinese_translation_source(item)
        for item in items
    ]

def _align_ocr_items_to_chinese_speech(
    items: list[dict],
    speech_windows: list[dict],
) -> list[dict]:
    if not items or not speech_windows:
        return items

    aligned: list[dict] = []
    previous_end = 0.0
    search_from = 0

    for item in items:
        item_start = float(item["start"])
        item_end = float(item["end"])
        item_center = (item_start + item_end) / 2.0

        best_index = None
        best_score = float("-inf")

        # Tìm cửa sổ giọng Trung gần và trùng thời gian nhất.
        upper = min(
            len(speech_windows),
            search_from + 10,
        )

        for index in range(search_from, upper):
            window = speech_windows[index]
            speech_start = float(window["start"])
            speech_end = float(window["end"])
            speech_center = (
                speech_start + speech_end
            ) / 2.0

            overlap = _time_overlap(
                item_start,
                item_end,
                speech_start,
                speech_end,
            )
            center_distance = abs(
                item_center - speech_center
            )

            score = overlap * 10.0 - center_distance

            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            aligned.append(dict(item))
            continue

        selected = speech_windows[best_index]
        speech_start = float(selected["start"])
        speech_end = float(selected["end"])

        # Không cho câu sau chạy ngược lên câu trước.
        speech_start = max(
            speech_start,
            previous_end,
        )
        speech_end = max(
            speech_start + 0.25,
            speech_end,
        )

        updated = dict(item)
        updated["start"] = speech_start
        updated["end"] = speech_end
        updated["speech_sync_source"] = "chinese_audio"
        updated["recognized_speech"] = str(
            selected.get("text") or ""
        )

        aligned.append(updated)

        previous_end = speech_end
        search_from = min(
            best_index + 1,
            len(speech_windows),
        )

    return aligned


def _probe_audio_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:"
            "nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    try:
        return max(
            0.01,
            float(result.stdout.strip()),
        )
    except Exception:
        return 0.01


def _atempo_chain(speed: float) -> str:
    speed = max(0.01, float(speed))
    factors: list[float] = []

    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0

    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5

    factors.append(speed)

    return ",".join(
        f"atempo={factor:.6f}"
        for factor in factors
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
        "FontSize=12,"
        "PrimaryColour=&H00FFFFFF,"
        "BackColour=&H90000000,"
        "OutlineColour=&H00000000,"
        "BorderStyle=3,"
        "Outline=0,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginL=22,"
        "MarginR=22,"
        "MarginV=22"
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
    voice_files: list[tuple[Path, int, int]],
    output_path: Path,
) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
    ]

    for voice_path, _, _ in voice_files:
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

    for index, (
        voice_path,
        delay_ms,
        target_duration_ms,
    ) in enumerate(
        voice_files,
        start=1,
    ):
        label = f"voice{index}"

        source_duration = _probe_audio_duration(
            voice_path
        )
        target_duration = max(
            0.25,
            float(target_duration_ms) / 1000.0,
        )

        # Nếu câu Việt dài hơn lượt nói Trung,
        # tăng tốc vừa đúng để không lấn sang câu kế tiếp.
        speed = max(
            1.0,
            source_duration / target_duration,
        )
        tempo_filter = _atempo_chain(speed)

        filters.append(
            f"[{index}:a]"
            f"aresample=44100,"
            f"{tempo_filter},"
            f"atrim=0:{target_duration:.6f},"
            f"asetpts=PTS-STARTPTS,"
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



def _cjk_count(value: str) -> int:
    return sum(
        1
        for char in value
        if "\u3400" <= char <= "\u9fff"
    )


def _normalise_ocr_text(value: str) -> str:
    value = " ".join(value.split()).strip()

    value = re.sub(
        r"[^\u3400-\u9fffA-Za-z0-9，。！？：；、]",
        "",
        value,
    )

    return value[:100]



def _is_valid_caption_text(value: str) -> bool:
    cjk = _cjk_count(value)

    if cjk < 2 or cjk > 42:
        return False

    if len(value) > 64:
        return False

    blocked = (
        "抖音",
        "快手",
        "关注",
        "点赞",
        "评论",
        "字幕组",
        "版权所有",
        "原创",
        "直播",
        "主页",
    )

    if any(token in value for token in blocked):
        return False

    if re.search(r"([\u3400-\u9fff])\1{3,}", value):
        return False

    cjk_ratio = cjk / max(1, len(value))

    if cjk_ratio < 0.42:
        return False

    return True

def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0

    return SequenceMatcher(
        None,
        left,
        right,
    ).ratio()


_RAPID_OCR_ENGINE = None


def _get_rapid_ocr():
    global _RAPID_OCR_ENGINE

    if _RAPID_OCR_ENGINE is not None:
        return _RAPID_OCR_ENGINE

    if RapidOCR is None:
        return None

    try:
        _RAPID_OCR_ENGINE = RapidOCR()
    except Exception:
        _RAPID_OCR_ENGINE = None

    return _RAPID_OCR_ENGINE


def _rapid_ocr_candidate(image) -> str:
    engine = _get_rapid_ocr()

    if engine is None:
        return ""

    try:
        result, _ = engine(image)
    except Exception:
        return ""

    if not result:
        return ""

    candidates = []

    for entry in result:
        if not entry or len(entry) < 3:
            continue

        raw_text = _normalise_ocr_text(
            str(entry[1] or "")
        )

        try:
            confidence = float(entry[2])
        except Exception:
            confidence = 0.0

        cjk = _cjk_count(raw_text)

        if cjk < 2:
            continue

        if confidence < 0.30:
            continue

        if not _is_valid_caption_text(raw_text):
            continue

        candidates.append({
            "text": raw_text,
            "confidence": confidence,
            "cjk": cjk,
        })

    if not candidates:
        return ""

    candidates.sort(
        key=lambda item: (
            item["confidence"],
            item["cjk"],
            len(item["text"]),
        ),
        reverse=True,
    )

    return candidates[0]["text"]


def _tesseract_fallback_candidate(image) -> str:
    try:
        raw = pytesseract.image_to_string(
            image,
            lang="chi_sim",
            config="--psm 7",
        )
    except Exception:
        return ""

    candidate = _normalise_ocr_text(raw)

    if not _is_valid_caption_text(candidate):
        return ""

    return candidate


def _ocr_one_frame(frame_path: Path) -> str:
    image = cv2.imread(str(frame_path))

    if image is None:
        return ""

    height, width = image.shape[:2]

    # Video đã xác định:
    # chữ Trung nằm ở dòng sát đáy.
    top = max(0, int(height * 0.915))
    bottom = min(height, int(height * 0.999))
    left = max(0, int(width * 0.02))
    right = min(width, int(width * 0.98))

    crop = image[top:bottom, left:right]

    if crop.size == 0:
        return ""

    enlarged = cv2.resize(
        crop,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )

    # Tăng nét chữ trắng có viền đen.
    blurred = cv2.GaussianBlur(
        enlarged,
        (0, 0),
        1.2,
    )

    sharpened = cv2.addWeighted(
        enlarged,
        1.8,
        blurred,
        -0.8,
        0,
    )

    rapid_candidates = [
        _rapid_ocr_candidate(enlarged),
        _rapid_ocr_candidate(sharpened),
    ]

    rapid_candidates = [
        value
        for value in rapid_candidates
        if _is_valid_caption_text(value)
    ]

    if rapid_candidates:
        return max(
            rapid_candidates,
            key=lambda value: (
                _cjk_count(value),
                len(value),
            ),
        )

    # Chỉ dùng Tesseract khi RapidOCR không cho kết quả.
    gray = cv2.cvtColor(
        sharpened,
        cv2.COLOR_BGR2GRAY,
    )

    threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )[1]

    fallback_candidates = [
        _tesseract_fallback_candidate(gray),
        _tesseract_fallback_candidate(threshold),
    ]

    fallback_candidates = [
        value
        for value in fallback_candidates
        if _is_valid_caption_text(value)
    ]

    if not fallback_candidates:
        return ""

    return max(
        fallback_candidates,
        key=lambda value: (
            _cjk_count(value),
            len(value),
        ),
    )


def _extract_ocr_items(
    input_path: Path,
    job_dir: Path,
) -> list[dict]:
    frames_dir = job_dir / "ocr_frames"

    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_pattern = (
        frames_dir / "frame_%06d.png"
    )

    _run([
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        (
            "fps=1,scale=iw:ih"
        ),
        str(frame_pattern),
    ])

    frames = sorted(
        frames_dir.glob("frame_*.png")
    )

    samples = []

    for index, frame_path in enumerate(
        frames
    ):
        # fps=1 nên mỗi khung cách nhau đúng 1 giây.
        timestamp = index * 1.0
        detected = _ocr_one_frame(frame_path)

        if not _is_valid_caption_text(detected):
            continue

        samples.append({
            "time": timestamp,
            "text": detected,
        })

    if not samples:
        return []

    groups = []

    for sample in samples:
        if not groups:
            groups.append({
                "start": sample["time"],
                "last": sample["time"],
                "texts": [sample["text"]],
            })
            continue

        current = groups[-1]

        representative = max(
            current["texts"],
            key=lambda value: (
                _cjk_count(value),
                len(value),
            ),
        )

        sample_text = sample["text"]

        similarity = _text_similarity(
            representative,
            sample_text,
        )

        gap = (
            sample["time"]
            - current["last"]
        )

        representative_chars = set(
            char for char in representative
            if "\u4e00" <= char <= "\u9fff"
        )

        sample_chars = set(
            char for char in sample_text
            if "\u4e00" <= char <= "\u9fff"
        )

        smaller_count = min(
            len(representative_chars),
            len(sample_chars),
        )

        shared_ratio = (
            len(
                representative_chars
                & sample_chars
            )
            / smaller_count
            if smaller_count
            else 0.0
        )

        same_caption = (
            similarity >= 0.42
            or (
                smaller_count >= 3
                and shared_ratio >= 0.55
            )
            or representative in sample_text
            or sample_text in representative
        )

        # fps=1: cho phép một khung OCR bị hụt,
        # nhưng không nối các câu cách nhau quá xa.
        if same_caption and gap <= 1.6:
            current["last"] = sample["time"]
            current["texts"].append(
                sample_text
            )
        else:
            groups.append({
                "start": sample["time"],
                "last": sample["time"],
                "texts": [sample_text],
            })

    items = []

    for index, group in enumerate(groups):
        source = max(
            group["texts"],
            key=lambda value: (
                _cjk_count(value),
                len(value),
            ),
        )

        if not _is_valid_caption_text(source):
            continue

        if index + 1 < len(groups):
            next_start = float(
                groups[index + 1]["start"]
            )
        else:
            next_start = (
                float(group["last"]) + 1.0
            )

        start = float(group["start"])

        end = min(
            next_start,
            max(
                start + 0.8,
                float(group["last"]) + 0.7,
            ),
        )

        if end - start > 4.8:
            end = start + 4.8

        if (
            items
            and _text_similarity(
                items[-1]["source"],
                source,
            ) >= 0.88
        ):
            items[-1]["end"] = max(
                items[-1]["end"],
                end,
            )
            continue

        items.append({
            "start": start,
            "end": end,
            "source": source,
        })

    return items

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
            "recognition_primary": "chinese_caption_ocr",
            "recognition_fallback": "whisper",
            "diagnostic_endpoint": "/autocap/diagnose",
            "ocr_band_mode": "rapidocr_exact_bottom_line",
            "ocr_sampling_fps": 1,
            "ocr_candidate_bands": 1,
            "ocr_vertical_range": "91.5%-99.9%",
            "ocr_horizontal_range": "4%-96%",
            "ocr_language": "chi_sim",
            "ocr_engine_primary": "rapidocr_onnxruntime",
            "ocr_engine_fallback": "tesseract_chi_sim",
            "ocr_upscale_factor": 4.0,
            "vietnamese_line_excluded": True,
            "ocr_text_line_count": 1,
            "ocr_caption_stabilizer": True,
            "ocr_frame_interval_seconds": 1.0,
            "ocr_group_max_gap_seconds": 1.6,
            "subtitle_timing_source": "chinese_speech",
            "dubbing_timing_source": "chinese_speech",
            "translation_source_policy": "stabilized_ocr_only",
            "ocr_whisper_crosscheck": False,
            "mismatched_ocr_replacement": False,
            "one_voice_file_per_speech_turn": True,
            "voice_overlap_prevention": True,
            "voices": {
                "female": "vi-VN-HoaiMyNeural",
                "male": "vi-VN-NamMinhNeural",
            },
            "max_upload_mb": 120,
        }

    @app.post("/autocap/diagnose")
    def autocap_diagnose(
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
                tempfile.mkdtemp(
                    prefix="nami_autocap_diagnostic_"
                )
            )

            input_path = job_dir / "input.mp4"

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

                items = _extract_ocr_items(
                    input_path,
                    job_dir,
                )

                recognition_source = "ocr"

                if not items:
                    recognition_source = (
                        "whisper_fallback"
                    )

                    if not _video_has_audio(
                        input_path
                    ):
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "OCR không đọc được chữ "
                                "và video không có âm thanh."
                            ),
                        )

                    audio_path = job_dir / "audio.wav"

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
                        condition_on_previous_text=False,
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
                            "nội dung tiếng Trung."
                        ),
                    )

                translations = _translate_batch([
                    item["source"]
                    for item in items
                ])

                diagnostic_items = []

                for index, item in enumerate(
                    items,
                    start=1,
                ):
                    source = str(
                        item.get("source", "")
                    ).strip()

                    vietnamese = str(
                        translations[index - 1]
                        if index - 1 < len(translations)
                        else ""
                    ).strip()

                    start = float(
                        item.get("start", 0.0)
                    )

                    end = float(
                        item.get("end", start)
                    )

                    diagnostic_items.append({
                        "index": index,
                        "start_seconds": round(
                            start,
                            3,
                        ),
                        "end_seconds": round(
                            end,
                            3,
                        ),
                        "duration_seconds": round(
                            max(0.0, end - start),
                            3,
                        ),
                        "start_srt": _timestamp(
                            start
                        ),
                        "end_srt": _timestamp(
                            end
                        ),
                        "source_zh": source,
                        "translation_vi": vietnamese,
                        "cjk_count": _cjk_count(
                            source
                        ),
                        "source_length": len(
                            source
                        ),
                        "valid_caption": (
                            _is_valid_caption_text(
                                source
                            )
                            if recognition_source == "ocr"
                            else True
                        ),
                    })

                return {
                    "ok": True,
                    "version": VERSION,
                    "filename": video.filename,
                    "recognition_source": (
                        recognition_source
                    ),
                    "item_count": len(
                        diagnostic_items
                    ),
                    "rendered_video": False,
                    "dubbing_created": False,
                    "items": diagnostic_items,
                }

            finally:
                shutil.rmtree(
                    job_dir,
                    ignore_errors=True,
                )

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

                items = _extract_ocr_items(
                    input_path,
                    job_dir,
                )

                recognition_source = "ocr"

                if len(items) < 2:
                    recognition_source = (
                        "whisper_fallback"
                    )

                    if not _video_has_audio(
                        input_path
                    ):
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "OCR không đọc được "
                                "phụ đề Trung và video "
                                "không có âm thanh."
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
                        condition_on_previous_text=False,
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
                            "Không đọc được "
                            "phụ đề tiếng Trung."
                        ),
                    )

                # V147I:
                # OCR chỉ quyết định nội dung chữ Trung.
                # Tiếng nói Trung quyết định thời điểm bắt đầu/kết thúc.
                if (
                    recognition_source == "ocr"
                    and _video_has_audio(input_path)
                ):
                    _extract_audio_for_sync(
                        input_path,
                        audio_path,
                    )

                    speech_windows = (
                        _detect_chinese_speech_windows(
                            audio_path
                        )
                    )

                    if speech_windows:
                        items = (
                            _align_ocr_items_to_chinese_speech(
                                items,
                                speech_windows,
                            )
                        )

                # V147M:
                # OCR ổn định quyết định nội dung cần dịch.
                # Whisper chỉ hỗ trợ xác định thời điểm nói,
                # tuyệt đối không thay câu chữ OCR.
                for item in items:
                    item["translation_source"] = str(
                        item.get("source", "")
                    ).strip()

                translations = _translate_batch([
                    item["translation_source"]
                    for item in items
                ])

                compact_items = []

                for item, vietnamese in zip(
                    items,
                    translations,
                ):
                    vietnamese = " ".join(
                        str(vietnamese or "").split()
                    ).strip()

                    if not vietnamese:
                        vietnamese = (
                            "[Không dịch được câu này]"
                        )

                    words = vietnamese.split()
                    chunks = []
                    current = ""

                    for word in words:
                        candidate = (
                            word
                            if not current
                            else current + " " + word
                        )

                        if len(candidate) <= 58:
                            current = candidate
                        else:
                            if current:
                                chunks.append(current)
                            current = word

                    if current:
                        chunks.append(current)

                    if not chunks:
                        continue

                    original_start = float(item["start"])
                    original_end = float(item["end"])
                    original_duration = max(
                        1.2,
                        original_end - original_start,
                    )

                    chunk_duration = max(
                        1.2,
                        min(
                            4.5,
                            original_duration
                            / len(chunks),
                        ),
                    )

                    for chunk_index, chunk in enumerate(
                        chunks
                    ):
                        chunk_start = (
                            original_start
                            + chunk_index * chunk_duration
                        )

                        chunk_end = min(
                            original_end,
                            chunk_start + chunk_duration,
                        )

                        if chunk_end <= chunk_start:
                            chunk_end = chunk_start + 1.2

                        compact_items.append({
                            "start": chunk_start,
                            "end": chunk_end,
                            "source": item["source"],
                            "vietnamese": chunk,
                        })

                items = compact_items

                subtitle_blocks = []

                for index, item in enumerate(
                    items,
                    start=1,
                ):
                    vietnamese = item["vietnamese"]
                    words = vietnamese.split()

                    lines = []
                    current_line = ""

                    for word in words:
                        candidate = (
                            word
                            if not current_line
                            else current_line + " " + word
                        )

                        if len(candidate) <= 31:
                            current_line = candidate
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word

                    if current_line:
                        lines.append(current_line)

                    if len(lines) > 2:
                        lines = [
                            lines[0],
                            " ".join(lines[1:]),
                        ]

                    display_text = "\\N".join(lines[:2])

                    subtitle_blocks.append(
                        f"{index}\n"
                        f'{_timestamp(item["start"])} '
                        f'--> {_timestamp(item["end"])}\n'
                        f"{display_text}\n"
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

                        start_ms = round(
                            float(item["start"]) * 1000
                        )
                        duration_ms = max(
                            250,
                            round(
                                (
                                    float(item["end"])
                                    - float(item["start"])
                                )
                                * 1000
                            ),
                        )

                        voice_files.append((
                            voice_path,
                            start_ms,
                            duration_ms,
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
