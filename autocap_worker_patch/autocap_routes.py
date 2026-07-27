from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
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

VERSION = "NAMI_V147Y_AUTO_SUBTITLE_BAND_DETECTION"
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

def _merge_chinese_caption_parts(
    current: str,
    incoming: str,
) -> str:
    left = "".join(
        str(current or "").split()
    ).strip()

    right = "".join(
        str(incoming or "").split()
    ).strip()

    if not left:
        return right

    if not right:
        return left

    if left == right:
        return left

    if right in left:
        return left

    if left in right:
        return right

    # Ghép phần đuôi dòng trước trùng với
    # phần đầu dòng sau để không lặp chữ.
    maximum_overlap = min(
        len(left),
        len(right),
        12,
    )

    for size in range(
        maximum_overlap,
        0,
        -1,
    ):
        if left[-size:] == right[:size]:
            return left + right[size:]

    # Hai dòng khác nhau nhưng cùng lượt nói:
    # nối bằng dấu phẩy Trung để bộ dịch hiểu
    # đây là một câu liên tục.
    return left.rstrip("，。！？") + "，" + right


def _align_ocr_items_to_chinese_speech(
    items: list[dict],
    speech_windows: list[dict],
) -> list[dict]:
    if not items or not speech_windows:
        return items

    assignments: list[dict] = []
    last_window_index = 0

    for item in items:
        item_start = float(
            item.get("start", 0.0)
        )
        item_end = float(
            item.get("end", item_start + 0.8)
        )
        item_center = (
            item_start + item_end
        ) / 2.0

        best_index = None
        best_score = float("-inf")

        # Cho phép nhiều dòng OCR cùng trỏ vào
        # một cửa sổ giọng nói. Không còn bắt buộc
        # mỗi dòng phải chuyển sang cửa sổ kế tiếp.
        search_start = max(
            0,
            last_window_index - 2,
        )

        search_end = min(
            len(speech_windows),
            last_window_index + 14,
        )

        for index in range(
            search_start,
            search_end,
        ):
            window = speech_windows[index]

            speech_start = float(
                window.get("start", 0.0)
            )
            speech_end = float(
                window.get(
                    "end",
                    speech_start + 0.5,
                )
            )
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

            # Ưu tiên giao nhau thật; vẫn cho phép
            # OCR lệch nhẹ tối đa khoảng một giây.
            score = (
                overlap * 12.0
                - center_distance
            )

            if (
                overlap <= 0
                and center_distance > 1.25
            ):
                score -= 8.0

            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            assignments.append({
                "window_index": None,
                "item": dict(item),
            })
            continue

        assignments.append({
            "window_index": best_index,
            "item": dict(item),
        })

        last_window_index = max(
            last_window_index,
            best_index,
        )

    merged: list[dict] = []

    for assignment in assignments:
        window_index = assignment[
            "window_index"
        ]
        item = assignment["item"]

        if window_index is None:
            merged.append(item)
            continue

        window = speech_windows[
            window_index
        ]

        speech_start = float(
            window.get("start", item["start"])
        )
        speech_end = float(
            window.get("end", item["end"])
        )

        source = str(
            item.get("source", "")
        ).strip()

        # Nếu dòng hiện tại và dòng trước cùng nằm
        # trong một cửa sổ giọng Trung, ghép thành
        # một lượt thoại duy nhất.
        if (
            merged
            and merged[-1].get(
                "_speech_window_index"
            ) == window_index
        ):
            previous = merged[-1]

            previous["source"] = (
                _merge_chinese_caption_parts(
                    previous.get("source", ""),
                    source,
                )
            )

            previous["start"] = min(
                float(previous["start"]),
                speech_start,
            )

            previous["end"] = max(
                float(previous["end"]),
                speech_end,
            )

            previous[
                "recognized_speech"
            ] = str(
                window.get("text") or ""
            )

            continue

        updated = dict(item)

        updated["start"] = speech_start
        updated["end"] = max(
            speech_start + 0.35,
            speech_end,
        )
        updated["source"] = source
        updated[
            "speech_sync_source"
        ] = "chinese_audio_turn"
        updated[
            "recognized_speech"
        ] = str(
            window.get("text") or ""
        )
        updated[
            "_speech_window_index"
        ] = window_index

        merged.append(updated)

    # Dọn trường nội bộ trước khi dịch.
    cleaned: list[dict] = []

    for item in merged:
        updated = dict(item)
        updated.pop(
            "_speech_window_index",
            None,
        )

        if not str(
            updated.get("source", "")
        ).strip():
            continue

        cleaned.append(updated)

    return cleaned


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

def _translation_is_valid(
    source: str,
    translated: str,
) -> bool:
    value = " ".join(
        str(translated or "").split()
    ).strip()

    if not value:
        return False

    if value == source.strip():
        return False

    lowered = value.lower()

    rejected_fragments = (
        "[không dịch được",
        "error 500",
        "error 502",
        "error 503",
        "error 504",
        "server error",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "please try again",
        "try again later",
        "temporarily unavailable",
        "too many requests",
        "access denied",
        "captcha",
        "<html",
        "<!doctype",
        "<body",
        "</html>",
        "cloudflare",
        "nginx",
    )

    if any(
        fragment in lowered
        for fragment in rejected_fragments
    ):
        return False

    if lowered.startswith(
        ("http://", "https://")
    ):
        return False

    # Phản hồi lỗi máy chủ thường dài và gần như
    # hoàn toàn là tiếng Anh, không phải câu Việt.
    latin_letters = sum(
        char.isascii() and char.isalpha()
        for char in value
    )

    vietnamese_marks = sum(
        char in (
            "ăâđêôơư"
            "áàảãạ"
            "ắằẳẵặ"
            "ấầẩẫậ"
            "éèẻẽẹ"
            "ếềểễệ"
            "íìỉĩị"
            "óòỏõọ"
            "ốồổỗộ"
            "ớờởỡợ"
            "úùủũụ"
            "ứừửữự"
            "ýỳỷỹỵ"
            "ĂÂĐÊÔƠƯ"
            "ÁÀẢÃẠ"
            "ẮẰẲẴẶ"
            "ẤẦẨẪẬ"
            "ÉÈẺẼẸ"
            "ẾỀỂỄỆ"
            "ÍÌỈĨỊ"
            "ÓÒỎÕỌ"
            "ỐỒỔỖỘ"
            "ỚỜỞỠỢ"
            "ÚÙỦŨỤ"
            "ỨỪỬỮỰ"
            "ÝỲỶỸỴ"
        )
        for char in value
    )

    if (
        len(value) > 45
        and latin_letters > 35
        and vietnamese_marks == 0
    ):
        return False

    return True


def _translate_batch(
    texts: list[str],
) -> list[str]:
    sources = [
        " ".join(str(value or "").split()).strip()
        for value in texts
    ]

    results = ["" for _ in sources]
    batch_size = 10

    # Dịch theo cụm nhỏ để tránh một lỗi mạng
    # làm trống toàn bộ 72 câu.
    for batch_start in range(
        0,
        len(sources),
        batch_size,
    ):
        indexes = list(range(
            batch_start,
            min(
                len(sources),
                batch_start + batch_size,
            ),
        ))

        pending = [
            index
            for index in indexes
            if sources[index]
        ]

        for attempt in range(4):
            if not pending:
                break

            source_language = (
                "zh-CN"
                if attempt < 2
                else "auto"
            )

            translator = GoogleTranslator(
                source=source_language,
                target="vi",
            )

            batch_texts = [
                sources[index]
                for index in pending
            ]

            translated_values = None

            try:
                translated_values = (
                    translator.translate_batch(
                        batch_texts
                    )
                )
            except Exception:
                translated_values = None

            next_pending = []

            if (
                isinstance(translated_values, list)
                and len(translated_values)
                == len(pending)
            ):
                for index, translated in zip(
                    pending,
                    translated_values,
                ):
                    value = " ".join(
                        str(translated or "").split()
                    ).strip()

                    if _translation_is_valid(
                        sources[index],
                        value,
                    ):
                        results[index] = value
                    else:
                        next_pending.append(index)
            else:
                next_pending = list(pending)

            pending = next_pending

            if pending:
                time.sleep(
                    1.0 + attempt * 1.5
                )

        # Những câu cụm vẫn lỗi được dịch riêng
        # từng câu, tạo translator mới mỗi lượt.
        for index in list(pending):
            for attempt in range(5):
                source_language = (
                    "zh-CN"
                    if attempt < 3
                    else "auto"
                )

                try:
                    translator = GoogleTranslator(
                        source=source_language,
                        target="vi",
                    )

                    value = translator.translate(
                        sources[index]
                    )

                    value = " ".join(
                        str(value or "").split()
                    ).strip()
                except Exception:
                    value = ""

                if _translation_is_valid(
                    sources[index],
                    value,
                ):
                    results[index] = value
                    break

                time.sleep(
                    1.0 + attempt
                )

    failed_indexes = [
        index
        for index, value in enumerate(results)
        if sources[index]
        and not _translation_is_valid(
            sources[index],
            value,
        )
    ]

    if failed_indexes:
        failed_sources = [
            sources[index]
            for index in failed_indexes[:5]
        ]

        raise RuntimeError(
            "TRANSLATION_INCOMPLETE_AFTER_RETRIES: "
            + " | ".join(failed_sources)
        )

    return results


MAX_DUB_SPEED = 1.10
MIN_DUB_TURN_SECONDS = 1.25
DUB_GAP_SECONDS = 0.10


def _compact_vietnamese_dub_text(
    value: str,
) -> str:
    text = " ".join(
        str(value or "").split()
    ).strip()

    replacements = [
        (
            "Dường như có một thây ma đang",
            "Hình như zombie đang",
        ),
        (
            "Có vẻ như có một thây ma đang",
            "Hình như zombie đang",
        ),
        (
            "một thây ma",
            "zombie",
        ),
        (
            "Cánh cửa này không chắc chắn",
            "Cửa không chắc đâu",
        ),
        (
            "Cánh cửa không chắc chắn",
            "Cửa không chắc đâu",
        ),
        (
            "Sao bạn dám đi bộ ở hành lang "
            "vào đêm khuya thế này?",
            "Muộn vậy còn dám đi ngoài hành lang?",
        ),
        (
            "Bây giờ chúng tôi dự định tập hợp "
            "tất cả những người sống sót vào "
            "đơn vị của chúng tôi",
            "Giờ tập hợp mọi người sống sót trong khu",
        ),
        (
            "Không có tiếng ồn lớn sẽ được "
            "thực hiện trong toàn bộ quá trình",
            "Chúng tôi sẽ không gây tiếng động lớn",
        ),
        (
            "Chúng tôi chỉ cần đăng ký số lượng "
            "người và dự trữ thực phẩm",
            "Chỉ cần ghi số người và lương thực",
        ),
        (
            "Không thể loại trừ khả năng bị cướp vật tư",
            "Có thể họ sẽ cướp vật tư",
        ),
        (
            "Họ không sợ thu hút zombie sao?",
            "Họ không sợ dụ zombie tới sao?",
        ),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    text = text.replace(
        "Dường như ",
        "Hình như ",
    )

    text = text.replace(
        "Sẽ thu hút nhiều zombie hơn",
        "Sẽ dụ thêm nhiều zombie",
    )

    return text.strip()


def _prepare_natural_dubbing_items(
    items: list[dict],
) -> list[dict]:
    prepared: list[dict] = []

    for index, original in enumerate(items):
        item = dict(original)

        start = float(
            item.get("start", 0.0)
        )

        source_end = float(
            item.get("end", start + 1.2)
        )

        vietnamese = _compact_vietnamese_dub_text(
            item.get("vietnamese", "")
        )

        if not vietnamese:
            continue

        word_count = max(
            1,
            len(vietnamese.split()),
        )

        # Thời lượng nói tự nhiên, nhưng câu vẫn
        # bắt đầu đúng lúc dòng chữ Trung xuất hiện.
        estimated_duration = max(
            1.20,
            word_count / 2.8,
        )

        # Không cho một câu kéo dài vô hạn.
        estimated_duration = min(
            5.2,
            estimated_duration,
        )

        if index + 1 < len(items):
            next_start = float(
                items[index + 1].get(
                    "start",
                    source_end + 2.0,
                )
            )

            available = max(
                0.35,
                next_start - start - 0.05,
            )

            # Khi đủ khoảng trống thì dùng trọn thời
            # lượng tự nhiên. Khi thiếu, bộ chỉnh tốc
            # chỉ tăng nhẹ tối đa 1.10 lần.
            desired_duration = max(
                source_end - start,
                min(
                    estimated_duration,
                    max(
                        available,
                        source_end - start,
                    ),
                ),
            )
        else:
            desired_duration = max(
                source_end - start,
                estimated_duration,
            )

        item["start"] = start
        item["end"] = (
            start
            + max(0.35, desired_duration)
        )
        item["vietnamese"] = vietnamese

        prepared.append(item)

    return prepared


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


async def _create_voice_batch(
    jobs: list[tuple[str, Path, str]],
    concurrency: int = 4,
) -> None:
    semaphore = asyncio.Semaphore(
        max(1, concurrency)
    )

    async def run_one(
        text: str,
        output_path: Path,
        voice: str,
    ) -> None:
        async with semaphore:
            last_error = None

            for attempt in range(3):
                try:
                    await _create_voice(
                        text,
                        output_path,
                        voice,
                    )

                    if (
                        output_path.is_file()
                        and output_path.stat().st_size > 0
                    ):
                        return

                    raise RuntimeError(
                        "TTS output is empty"
                    )
                except Exception as exc:
                    last_error = exc

                    await asyncio.sleep(
                        1.0 + attempt * 1.5
                    )

            raise RuntimeError(
                "Vietnamese TTS failed after retries: "
                + str(last_error)
            )

    await asyncio.gather(*[
        run_one(
            text,
            output_path,
            voice,
        )
        for text, output_path, voice in jobs
    ])


def _render_voice_batch_track(
    voice_files: list[tuple[Path, int, int]],
    output_path: Path,
    total_duration: float,
) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-t",
        f"{total_duration:.6f}",
        "-i",
        "anullsrc=channel_layout=stereo:"
        "sample_rate=44100",
    ]

    for voice_path, _, _ in voice_files:
        args.extend([
            "-i",
            str(voice_path),
        ])

    filters: list[str] = []
    mix_labels = ["[0:a]"]

    for index, (
        voice_path,
        delay_ms,
        target_duration_ms,
    ) in enumerate(
        voice_files,
        start=1,
    ):
        source_duration = _probe_audio_duration(
            voice_path
        )

        target_duration = max(
            0.25,
            float(target_duration_ms) / 1000.0,
        )

        requested_speed = max(
            1.0,
            source_duration / target_duration,
        )

        # Không ép giọng nhanh quá mức nghe được.
        speed = min(
            MAX_DUB_SPEED,
            requested_speed,
        )

        tempo_filter = _atempo_chain(speed)

        # V147V: tuyệt đối không cắt đuôi câu.
        # Giữ toàn bộ thời lượng sau khi chỉnh tốc độ.
        rendered_duration = (
            source_duration / speed
        )
        label = f"voice{index}"

        filters.append(
            f"[{index}:a]"
            f"aresample=44100,"
            f"{tempo_filter},"
            f"apad=pad_dur=0.05,"
            f"asetpts=PTS-STARTPTS,"
            f"adelay={delay_ms}|{delay_ms},"
            f"volume=1.35"
            f"[{label}]"
        )

        mix_labels.append(f"[{label}]")

    filters.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:"
        + "duration=first:"
        + "dropout_transition=0:"
        + "normalize=0"
        + "[batchmixed]"
    )

    _run([
        *args,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[batchmixed]",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-t",
        f"{total_duration:.6f}",
        str(output_path),
    ])


def _render_with_dubbing(
    input_path: Path,
    srt_path: Path,
    voice_files: list[tuple[Path, int, int]],
    output_path: Path,
) -> None:
    if not voice_files:
        _render_subtitle_only(
            input_path,
            srt_path,
            output_path,
        )
        return

    total_duration = max(
        0.5,
        _probe_audio_duration(input_path),
    )

    batch_size = 10
    batch_tracks: list[Path] = []

    for batch_index, start in enumerate(
        range(0, len(voice_files), batch_size),
        start=1,
    ):
        batch = voice_files[
            start:start + batch_size
        ]

        batch_path = (
            output_path.parent
            / f"dub_batch_{batch_index:03d}.m4a"
        )

        _render_voice_batch_track(
            batch,
            batch_path,
            total_duration,
        )

        batch_tracks.append(batch_path)

        for voice_path, _, _ in batch:
            try:
                voice_path.unlink()
            except Exception:
                pass

    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
    ]

    for batch_path in batch_tracks:
        args.extend([
            "-i",
            str(batch_path),
        ])

    filters: list[str] = []
    mix_labels: list[str] = []

    if _video_has_audio(input_path):
        filters.append(
            "[0:a]"
            "volume=0.14,"
            "aresample=44100"
            "[original]"
        )
    else:
        filters.append(
            "anullsrc=channel_layout=stereo:"
            "sample_rate=44100"
            "[original]"
        )

    mix_labels.append("[original]")

    for index in range(
        1,
        len(batch_tracks) + 1,
    ):
        label = f"batch{index}"

        filters.append(
            f"[{index}:a]"
            f"aresample=44100"
            f"[{label}]"
        )

        mix_labels.append(f"[{label}]")

    filters.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:"
        + "duration=first:"
        + "dropout_transition=0:"
        + "normalize=0"
        + "[mixed]"
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

    for batch_path in batch_tracks:
        try:
            batch_path.unlink()
        except Exception:
            pass



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

    # Tự kiểm tra nhiều vị trí phụ đề.
    # Vùng đầu tiên giữ nguyên video cũ.
    subtitle_bands = [
        {
            "name": "exact_bottom",
            "top": 0.915,
            "bottom": 0.999,
            "priority": 3,
        },
        {
            "name": "lower_middle",
            "top": 0.68,
            "bottom": 0.90,
            "priority": 2,
        },
        {
            "name": "middle",
            "top": 0.52,
            "bottom": 0.78,
            "priority": 1,
        },
    ]

    rapid_results: list[dict] = []
    prepared_bands: list[dict] = []

    for band in subtitle_bands:
        top = max(
            0,
            int(height * float(band["top"])),
        )
        bottom = min(
            height,
            int(height * float(band["bottom"])),
        )
        left = max(
            0,
            int(width * 0.02),
        )
        right = min(
            width,
            int(width * 0.98),
        )

        crop = image[
            top:bottom,
            left:right,
        ]

        if crop.size == 0:
            continue

        enlarged = cv2.resize(
            crop,
            None,
            fx=4.0,
            fy=4.0,
            interpolation=cv2.INTER_CUBIC,
        )

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

        prepared_bands.append({
            "name": band["name"],
            "priority": band["priority"],
            "enlarged": enlarged,
            "sharpened": sharpened,
        })

        # Trước tiên chỉ chạy một lượt RapidOCR
        # cho mỗi vùng để tránh tăng thời gian quá mạnh.
        value = _rapid_ocr_candidate(
            enlarged
        )

        if _is_valid_caption_text(value):
            rapid_results.append({
                "text": value,
                "band": band["name"],
                "priority": band["priority"],
                "quality": (
                    _is_high_quality_chinese_caption(
                        value
                    )
                ),
            })

    high_quality_rapid = [
        result
        for result in rapid_results
        if result["quality"]
    ]

    if high_quality_rapid:
        selected = max(
            high_quality_rapid,
            key=lambda result: (
                _cjk_count(result["text"]),
                len(result["text"]),
                result["priority"],
            ),
        )

        return str(selected["text"])

    # Nếu ảnh gốc chưa đọc tốt, thử ảnh tăng nét.
    sharpened_results: list[dict] = []

    for prepared in prepared_bands:
        value = _rapid_ocr_candidate(
            prepared["sharpened"]
        )

        if not _is_valid_caption_text(value):
            continue

        sharpened_results.append({
            "text": value,
            "band": prepared["name"],
            "priority": prepared["priority"],
            "quality": (
                _is_high_quality_chinese_caption(
                    value
                )
            ),
        })

    high_quality_sharpened = [
        result
        for result in sharpened_results
        if result["quality"]
    ]

    if high_quality_sharpened:
        selected = max(
            high_quality_sharpened,
            key=lambda result: (
                _cjk_count(result["text"]),
                len(result["text"]),
                result["priority"],
            ),
        )

        return str(selected["text"])

    # Tesseract chỉ chạy khi RapidOCR ở tất cả
    # các vùng đều không tìm được câu Trung tốt.
    fallback_results: list[dict] = []

    for prepared in prepared_bands:
        gray = cv2.cvtColor(
            prepared["sharpened"],
            cv2.COLOR_BGR2GRAY,
        )

        threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )[1]

        for processed in (
            gray,
            threshold,
        ):
            value = (
                _tesseract_fallback_candidate(
                    processed
                )
            )

            if not _is_valid_caption_text(value):
                continue

            fallback_results.append({
                "text": value,
                "priority": prepared["priority"],
                "quality": (
                    _is_high_quality_chinese_caption(
                        value
                    )
                ),
            })

    high_quality_fallback = [
        result
        for result in fallback_results
        if result["quality"]
    ]

    if high_quality_fallback:
        selected = max(
            high_quality_fallback,
            key=lambda result: (
                _cjk_count(result["text"]),
                len(result["text"]),
                result["priority"],
            ),
        )

        return str(selected["text"])

    # Không trả chữ rác hoặc watermark về pipeline.
    return ""


def _is_high_quality_chinese_caption(
    value: str,
) -> bool:
    text = _normalise_ocr_text(
        str(value or "")
    )

    if not _is_valid_caption_text(text):
        return False

    cjk = _cjk_count(text)

    latin_count = sum(
        char.isascii() and char.isalpha()
        for char in text
    )

    digit_count = sum(
        char.isdigit()
        for char in text
    )

    meaningful_count = sum(
        char.isalnum()
        or "\u4e00" <= char <= "\u9fff"
        for char in text
    )

    cjk_ratio = (
        cjk / meaningful_count
        if meaningful_count
        else 0.0
    )

    # RapidOCR đôi khi đọc chữ hiệu ứng hoặc watermark
    # thành chuỗi trộn chữ Latin như "An", "PK", "VIP".
    if latin_count > 0:
        return False

    # Loại các mảnh ngắn có số chen giữa như:
    # "及1一个上", "1怕二让人的".
    if digit_count > 0 and cjk < 6:
        return False

    if digit_count > 1:
        return False

    # Câu Trung bình thường phải chủ yếu là chữ Hán.
    if cjk_ratio < 0.78:
        return False

    # Vẫn giữ câu ngắn hợp lệ như 快走, 救命, 别开.
    if cjk < 2:
        return False

    return True


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

        if not _is_high_quality_chinese_caption(
            source
        ):
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
            "ocr_band_mode": "rapidocr_auto_multi_band",
            "ocr_sampling_fps": 1,
            "ocr_candidate_bands": 3,
            "ocr_vertical_range": "auto:52%-78%,68%-90%,91.5%-99.9%",
            "ocr_horizontal_range": "4%-96%",
            "ocr_language": "chi_sim",
            "ocr_engine_primary": "rapidocr_onnxruntime",
            "ocr_engine_fallback": "tesseract_chi_sim",
            "ocr_upscale_factor": 4.0,
            "ocr_garbage_filter": True,
            "reject_latin_mixed_text": True,
            "reject_short_digit_mixed_text": True,
            "minimum_cjk_ratio": 0.78,
            "vietnamese_line_excluded": True,
            "ocr_text_line_count": 1,
            "ocr_caption_stabilizer": True,
            "auto_subtitle_band_detection": True,
            "subtitle_band_per_frame_scoring": True,
            "exact_bottom_band_preserved": True,
            "lower_middle_subtitle_support": True,
            "middle_subtitle_support": True,
            "watermark_rejection_after_band_scan": True,
            "ocr_frame_interval_seconds": 1.0,
            "ocr_group_max_gap_seconds": 1.6,
            "subtitle_timing_source": "chinese_speech",
            "dubbing_timing_source": "chinese_speech",
            "translation_source_policy": "stabilized_ocr_only",
            "ocr_whisper_crosscheck": False,
            "mismatched_ocr_replacement": False,
            "one_voice_file_per_speech_turn": True,
            "voice_overlap_prevention": True,
            "dubbing_batch_size": 10,
            "tts_batch_size": 8,
            "tts_concurrency": 4,
            "tts_retry_count": 3,
            "original_audio_volume": 0.14,
            "maximum_dubbing_speed": 1.35,
            "minimum_dubbing_turn_seconds": 1.25,
            "dubbing_gap_seconds": 0.10,
            "natural_vietnamese_compaction": True,
            "voice_tail_cut_prevention": True,
            "translation_batch_size": 10,
            "translation_batch_retry_count": 4,
            "translation_sentence_retry_count": 5,
            "translation_auto_fallback": True,
            "translation_placeholder_allowed": False,
            "natural_dubbing_helpers_restored": True,
            "missing_helper_runtime_guard": True,
            "reject_translation_server_errors": True,
            "reject_translation_html": True,
            "reject_translation_error_codes": [500, 502, 503, 504],
            "server_error_text_rendering_allowed": False,
            "maximum_dubbing_speed": 1.10,
            "no_voice_trimming": True,
            "sequential_voice_scheduling": True,
            "allow_voice_overrun": True,
            "minimum_voice_gap_seconds": 0.12,
            "one_chinese_caption_one_vietnamese_voice": True,
            "translation_chunk_splitting": False,
            "voice_fragment_splitting": False,
            "voice_start_locked_to_caption": True,
            "cumulative_voice_delay_prevention": True,
            "speech_turn_caption_merge": True,
            "multiple_ocr_lines_per_speech_turn": True,
            "caption_overlap_deduplication": True,
            "translation_after_speech_turn_merge": True,
            "one_voice_file_per_complete_turn": True,
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
                        raise RuntimeError(
                            "EMPTY_TRANSLATION_FOR_SOURCE: "
                            + str(
                                item.get("source", "")
                            )
                        )

                    # V147W:
                    # Một dòng phụ đề Trung tương ứng đúng
                    # một câu Việt hoàn chỉnh.
                    # Không chia câu Việt thành nhiều mẩu
                    # và không tạo nhiều file giọng cho
                    # cùng một câu Trung.
                    compact_items.append({
                        "start": float(
                            item.get("start", 0.0)
                        ),
                        "end": float(
                            item.get(
                                "end",
                                float(
                                    item.get("start", 0.0)
                                ) + 1.2,
                            )
                        ),
                        "source": str(
                            item.get("source", "")
                        ).strip(),
                        "vietnamese": vietnamese,
                    })

                items = compact_items

                # V147R: trước khi tạo phụ đề và giọng,
                # nới lượt thoại vào khoảng trống,
                # rút gọn câu dài và giới hạn tốc độ.
                if mode == "dub":
                    items = (
                        _prepare_natural_dubbing_items(
                            items
                        )
                    )

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
                    tts_jobs = []

                    for index, item in enumerate(
                        items,
                        start=1,
                    ):
                        voice_path = (
                            job_dir
                            / f"voice_{index:04d}.mp3"
                        )

                        tts_jobs.append((
                            str(item["vietnamese"]),
                            voice_path,
                            voice,
                        ))

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

                    # Tạo giọng từng cụm nhỏ, tối đa
                    # bốn kết nối TTS chạy đồng thời.
                    tts_batch_size = 8

                    for start in range(
                        0,
                        len(tts_jobs),
                        tts_batch_size,
                    ):
                        asyncio.run(
                            _create_voice_batch(
                                tts_jobs[
                                    start:
                                    start + tts_batch_size
                                ],
                                concurrency=4,
                            )
                        )

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
