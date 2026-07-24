from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
import json

VERSION = "nami-worker-emergency-minimal-v1"

app = FastAPI(title="NC AI Money Worker - Emergency Minimal")

@app.get("/")
def root():
    return {
        "ok": True,
        "version": VERSION,
        "status": "WORKER_ALIVE_EMERGENCY_MINIMAL",
        "time": datetime.now().isoformat()
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "version": VERSION,
        "status": "HEALTH_OK",
        "time": datetime.now().isoformat(),
        "available_tasks": [
            "health",
            "hoathinhgau_2d_list_jobs",
            "hoathinhgau_2d_next_job",
            "worker_asset_license_review"
        ]
    }

def list_hoathinhgau_jobs():
    job_dir = Path("worker_jobs/hoathinhgau_2d_v41")
    jobs = []
    if job_dir.exists():
        for p in sorted(job_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["job_file"] = str(p)
                jobs.append(data)
            except Exception as e:
                jobs.append({"job_file": str(p), "status": "read_error", "error": str(e)})
    return jobs

def asset_license_review():
    seed = Path("nami_tools_2d/hoathinhgau_best_real_asset_seed_pack_v38.json")
    out_dir = Path("worker_outputs/hoathinhgau_2d_v49")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "license_review_report.json"

    if not seed.exists():
        return {
            "ok": False,
            "status": "MISSING_SEED_PACK",
            "missing": str(seed)
        }

    data = json.loads(seed.read_text(encoding="utf-8"))
    packages = data.get("source_packages", [])

    rows = []
    counts = {
        "private_test_ok": 0,
        "public_use_needs_manual_review": 0,
        "missing_source_url": 0,
        "total": 0
    }

    for p in packages:
        source_url = p.get("source_url")
        rows.append({
            "package_id": p.get("package_id"),
            "title": p.get("title"),
            "group": p.get("group"),
            "verified_kind": p.get("verified_kind"),
            "source_url": source_url,
            "private_test_status": "ok_for_private_pipeline_test",
            "public_use_status": "manual_license_review_required_before_public_release" if source_url else "blocked_missing_source_url"
        })
        counts["total"] += 1
        counts["private_test_ok"] += 1
        if source_url:
            counts["public_use_needs_manual_review"] += 1
        else:
            counts["missing_source_url"] += 1

    report = {
        "result": "PASS",
        "status": "NAMI_HOATHINHGAU_2D_ASSET_LICENSE_REVIEW_EMERGENCY_PASS",
        "created_at": datetime.now().isoformat(),
        "review_count": len(rows),
        "counts": counts,
        "rows": rows,
        "next_stage": "NAMI_2D_WORKER_RECOVERED_CONTINUE_SAFE"
    }

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "status": report["status"],
        "review_count": report["review_count"],
        "counts": counts,
        "output": str(out),
        "next_stage": report["next_stage"]
    }

@app.post("/run-job")
async def run_job(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    task = data.get("task") or data.get("job") or ""

    if task == "nami_worker_actor_005_layer_rig_plan_v97":
        return nami_worker_actor_005_layer_rig_plan_v97()

    if task == "nami_worker_learn_2d_film_v96":
        return nami_worker_learn_2d_film_v96()

    if task == "health":
        return health()

    if task in ["hoathinhgau_2d_list_jobs", "list_hoathinhgau_2d_jobs"]:
        jobs = list_hoathinhgau_jobs()
        return {
            "ok": True,
            "status": "NAMI_HOATHINHGAU_2D_JOBS_LISTED_EMERGENCY",
            "job_count": len(jobs),
            "jobs": jobs
        }

    if task in ["hoathinhgau_2d_next_job", "next_hoathinhgau_2d_job"]:
        jobs = list_hoathinhgau_jobs()
        queued = [j for j in jobs if j.get("status") == "queued"]
        queued.sort(key=lambda x: x.get("priority", 999))
        return {
            "ok": True,
            "status": "NAMI_HOATHINHGAU_2D_NEXT_JOB_READY_EMERGENCY",
            "queued_count": len(queued),
            "job": queued[0] if queued else None
        }

    if task in ["nami_2d_first_shot_asset_prepare_v51", "nami_2d_first_shot_asset_prepare"]:
        return nami_2d_first_shot_asset_prepare_v51()
    if task in ["nami_2d_first_shot_template_v52", "nami_2d_first_shot_template"]:
        return nami_2d_first_shot_template_v52()
    if task in ["nami_2d_first_shot_render_plan_v53", "nami_2d_first_shot_render_plan"]:
        return nami_2d_first_shot_render_plan_v53()
    if task in ["nami_2d_first_shot_private_render_v54", "nami_2d_first_shot_private_render"]:
        return nami_2d_first_shot_private_render_v54()
    if task in ["nami_2d_actor_talking_motion_v55", "nami_2d_actor_talking_motion"]:
        return nami_2d_actor_talking_motion_v55()
    if task in ["nami_2d_rig_learning_v56", "nami_2d_rig_learning"]:
        return nami_2d_rig_learning_v56()

    if task in ["worker_asset_license_review", "hoathinhgau_2d_asset_license_review"]:
        return asset_license_review()

    return JSONResponse(
        status_code=200,
        content={
            "ok": False,
            "error": "unknown_task",
            "task": task,
            "version": VERSION,
            "available_tasks": [
                "health",
                "hoathinhgau_2d_list_jobs",
                "hoathinhgau_2d_next_job",
                "worker_asset_license_review"
            ]
        }
    )


# === NAMI_2D_FIRST_SHOT_ASSET_PREPARE_V51 ===
def nami_2d_first_shot_asset_prepare_v51():
    import json
    from pathlib import Path
    from datetime import datetime

    seed = Path("nami_tools_2d/hoathinhgau_best_real_asset_seed_pack_v38.json")
    out_dir = Path("worker_outputs/nami_2d_first_shot_v51")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "first_shot_asset_plan_v51.json"

    if not seed.exists():
        return {
            "ok": False,
            "status": "NAMI_2D_FIRST_SHOT_MISSING_SEED_V51",
            "missing": str(seed)
        }

    data = json.loads(seed.read_text(encoding="utf-8"))
    packages = data.get("source_packages", [])
    previews = data.get("preview_images", [])

    def pick_pkg(role):
        for p in packages:
            if p.get("group") == role:
                return p
        return None

    def pick_preview(role):
        for p in previews:
            if p.get("role") == role:
                return p
        return None

    selected = {
        "character_package": pick_pkg("characters"),
        "background_package": pick_pkg("backgrounds_scenes"),
        "expression_package": pick_pkg("expressions"),
        "effect_package": pick_pkg("skills_effects"),
        "prop_package": pick_pkg("props_items"),
        "character_preview": pick_preview("characters"),
        "background_preview": pick_preview("backgrounds_scenes"),
        "expression_preview": pick_preview("expressions"),
        "effect_preview": pick_preview("skills_effects")
    }

    ready = selected["character_package"] is not None and selected["background_package"] is not None

    plan = {
        "result": "PASS",
        "status": "NAMI_2D_FIRST_SHOT_ASSET_PREPARE_V51_PASS",
        "created_at": datetime.now().isoformat(),
        "ready_for_first_shot_template": ready,
        "selected": selected,
        "shot_plan": {
            "duration_seconds": 8,
            "format": "9:16",
            "scene": "Một nhân vật đứng trong nền/cảnh, camera zoom nhẹ, biểu cảm đơn giản.",
            "animation_level": "first_test_simple_motion",
            "phone_role": "view_result_only",
            "worker_role": "download/extract/render later"
        },
        "next_stage": "NAMI_2D_FIRST_SHOT_TEMPLATE_V52"
    }

    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "status": plan["status"],
        "ready_for_first_shot_template": ready,
        "character_title": (selected["character_package"] or {}).get("title"),
        "background_title": (selected["background_package"] or {}).get("title"),
        "expression_title": (selected["expression_package"] or {}).get("title"),
        "effect_title": (selected["effect_package"] or {}).get("title"),
        "output": str(out),
        "next_stage": plan["next_stage"]
    }
# === END_NAMI_2D_FIRST_SHOT_ASSET_PREPARE_V51 ===



def nami_2d_first_shot_template_v52():
    import json
    from pathlib import Path
    from datetime import datetime

    out_dir = Path("worker_outputs/nami_2d_first_shot_v52")
    out_dir.mkdir(parents=True, exist_ok=True)

    shot = {
        "ok": True,
        "status": "NAMI_2D_FIRST_SHOT_TEMPLATE_V52_PASS",
        "created_at": datetime.utcnow().isoformat(),
        "shot_id": "first_2d_shot_001",
        "duration_seconds": 8,
        "format": "vertical_9_16",
        "resolution": "720x1280",
        "fps": 24,
        "scene": {
            "type": "private_test_only",
            "summary": "Nhân vật chính xuất hiện trong cảnh hiệu ứng sóng/khói, chuẩn bị cho shot hoạt hình 2D đầu tiên.",
            "safety": "private_test_only_no_public_upload_until_license_review"
        },
        "timeline": [
            {"time": "0.0-1.0", "action": "fade_in_background"},
            {"time": "1.0-2.5", "action": "character_enter_center"},
            {"time": "2.5-5.5", "action": "idle_motion_breathing_and_camera_push"},
            {"time": "5.5-7.2", "action": "effect_smoke_wave_burst"},
            {"time": "7.2-8.0", "action": "fade_out"}
        ],
        "next_stage": "NAMI_2D_FIRST_SHOT_RENDER_PLAN_V53"
    }

    out_file = out_dir / "first_shot_template_v52.json"
    out_file.write_text(json.dumps(shot, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**shot, "output": str(out_file)}


def nami_2d_first_shot_render_plan_v53():
    import json
    from pathlib import Path
    from datetime import datetime

    out_dir = Path("worker_outputs/nami_2d_first_shot_v53")
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "ok": True,
        "status": "NAMI_2D_FIRST_SHOT_RENDER_PLAN_V53_PASS",
        "created_at": datetime.utcnow().isoformat(),
        "render_target": {
            "duration_seconds": 8,
            "fps": 24,
            "frames": 192,
            "resolution": "720x1280",
            "format": "mp4_vertical_9_16"
        },
        "safe_worker_steps": [
            "load_or_generate_safe_placeholder_layers",
            "compose_background",
            "place_character_center",
            "animate_idle_motion",
            "animate_camera_push",
            "add_smoke_wave_effect",
            "export_private_test_mp4"
        ],
        "phone_allowed": False,
        "worker_required": True,
        "public_upload_allowed": False,
        "reason": "Asset license chưa duyệt public; chỉ render test riêng để kiểm tra pipeline.",
        "next_stage": "NAMI_2D_FIRST_SHOT_PRIVATE_RENDER_V54"
    }

    out_file = out_dir / "first_shot_render_plan_v53.json"
    out_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**plan, "output": str(out_file)}


def nami_2d_first_shot_private_render_v54():
    import json, math, subprocess, shutil
    from pathlib import Path
    from datetime import datetime

    out_dir = Path("worker_outputs/nami_2d_first_shot_v54")
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    width, height, fps, seconds = 720, 1280, 24, 8
    total = fps * seconds

    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        return {
            "ok": False,
            "status": "NAMI_2D_FIRST_SHOT_PRIVATE_RENDER_V54_BLOCKED",
            "reason": "Pillow/PIL missing on worker",
            "error": repr(e),
            "next_stage": "ADD_PILLOW_OR_USE_FFMPEG_ONLY_RENDER"
        }

    for i in range(total):
        t = i / max(1, total - 1)
        img = Image.new("RGB", (width, height), (8, 12, 24))
        d = ImageDraw.Draw(img)

        # background wave
        for y in range(0, height, 24):
            xoff = int(math.sin(t * 8 + y * 0.02) * 28)
            shade = 25 + int(25 * math.sin(t * 4 + y * 0.01))
            d.line([(0, y), (width, y + xoff)], fill=(shade, shade + 10, shade + 25), width=3)

        # camera push / character
        cx = width // 2
        cy = int(height * 0.56)
        scale = 1.0 + 0.08 * t
        body_w = int(150 * scale)
        body_h = int(280 * scale)
        breath = int(math.sin(t * math.pi * 8) * 6)

        # shadow
        d.ellipse((cx-130, cy+body_h//2+50, cx+130, cy+body_h//2+95), fill=(0,0,0))

        # character placeholder
        d.ellipse((cx-70, cy-body_h//2-120+breath, cx+70, cy-body_h//2+20+breath), fill=(225, 205, 170))
        d.rectangle((cx-body_w//2, cy-body_h//2+10+breath, cx+body_w//2, cy+body_h//2+breath), fill=(52, 96, 160))
        d.line((cx-body_w//2, cy-40+breath, cx-170, cy+80), fill=(225,205,170), width=18)
        d.line((cx+body_w//2, cy-40+breath, cx+170, cy+80), fill=(225,205,170), width=18)

        # smoke burst near end
        if t > 0.68:
            k = (t - 0.68) / 0.32
            for n in range(10):
                ang = n * 0.63
                r = int(40 + 260 * k + n * 4)
                sx = int(cx + math.cos(ang) * r)
                sy = int(cy + math.sin(ang) * r * 0.55)
                size = int(35 + 60 * (1-k))
                d.ellipse((sx-size, sy-size, sx+size, sy+size), outline=(180,190,210), width=4)

        # title
        d.text((28, 36), "NAMI 2D FIRST PRIVATE TEST", fill=(235,235,235))
        d.text((28, 72), "V54 - private pipeline render", fill=(180,190,210))

        img.save(frames_dir / f"frame_{i:04d}.png")

    mp4 = out_dir / "nami_first_2d_private_test_v54.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {
            "ok": False,
            "status": "NAMI_2D_FIRST_SHOT_PRIVATE_RENDER_V54_BLOCKED",
            "reason": "ffmpeg missing on worker",
            "frames_created": total,
            "frames_dir": str(frames_dir),
            "next_stage": "ADD_FFMPEG_TO_WORKER"
        }

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(mp4)
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if run.returncode != 0:
        return {
            "ok": False,
            "status": "NAMI_2D_FIRST_SHOT_PRIVATE_RENDER_V54_BLOCKED",
            "reason": "ffmpeg render failed",
            "stderr": run.stderr[-2000:],
            "next_stage": "FIX_RENDER_COMMAND"
        }

    report = {
        "ok": True,
        "status": "NAMI_2D_FIRST_SHOT_PRIVATE_RENDER_V54_PASS",
        "created_at": datetime.utcnow().isoformat(),
        "video": str(mp4),
        "frames": total,
        "duration_seconds": seconds,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "public_upload_allowed": False,
        "note": "Video test riêng tư để xác nhận pipeline render 2D hoạt động; chưa dùng để đăng công khai.",
        "next_stage": "NAMI_2D_FIRST_SHOT_QA_V55"
    }

    report_file = out_dir / "render_report_v54.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**report, "report": str(report_file)}


@app.get("/download/first-shot-v54")
def download_first_shot_v54():
    from pathlib import Path
    from fastapi.responses import FileResponse, JSONResponse

    f = Path("worker_outputs/nami_2d_first_shot_v54/nami_first_2d_private_test_v54.mp4")
    if not f.exists():
        return JSONResponse({
            "ok": False,
            "status": "FIRST_SHOT_V54_FILE_NOT_FOUND",
            "message": "Run task nami_2d_first_shot_private_render_v54 first."
        }, status_code=404)

    return FileResponse(
        str(f),
        media_type="video/mp4",
        filename="nami_first_2d_private_test_v54.mp4"
    )


def nami_2d_actor_talking_motion_v55():
    import json, math, subprocess, shutil
    from pathlib import Path
    from datetime import datetime

    out_dir = Path("worker_outputs/nami_2d_actor_talking_motion_v55")
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    actor_path = Path("assets/actors/owner_actor_v55.png")
    actor_b64_path = Path("assets/actors/owner_actor_v55.png.b64")

    if not actor_path.exists() and actor_b64_path.exists():
        import base64
        actor_path.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(actor_b64_path.read_text().encode())
        actor_path.write_bytes(raw)

    if not actor_path.exists():
        return {
            "ok": False,
            "status": "NAMI_2D_ACTOR_TALKING_MOTION_V55_BLOCKED",
            "reason": "actor file missing",
            "need": "assets/actors/owner_actor_v55.png or owner_actor_v55.png.b64"
        }

    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        return {
            "ok": False,
            "status": "NAMI_2D_ACTOR_TALKING_MOTION_V55_BLOCKED",
            "reason": "Pillow/PIL missing",
            "error": repr(e)
        }

    width, height, fps, seconds = 720, 1280, 24, 8
    total = fps * seconds

    actor = Image.open(actor_path).convert("RGBA")
    # fit actor to screen height
    target_h = 720
    ratio = target_h / actor.height
    target_w = max(1, int(actor.width * ratio))
    actor = actor.resize((target_w, target_h))

    for i in range(total):
        t = i / max(1, total - 1)

        bg = Image.new("RGB", (width, height), (13, 18, 30))
        d = ImageDraw.Draw(bg)

        # simple stage background
        for y in range(0, height, 32):
            shade = 30 + int(18 * math.sin(t * 5 + y * 0.015))
            d.line([(0, y), (width, y)], fill=(shade, shade + 6, shade + 18), width=2)

        # ground
        d.rectangle((0, 980, width, height), fill=(20, 24, 35))
        d.ellipse((210, 1030, 510, 1100), fill=(0, 0, 0))

        # actor motion: breathing + small walking sway
        bob = int(math.sin(t * math.pi * 8) * 10)
        sway = int(math.sin(t * math.pi * 4) * 14)
        x = (width - actor.width) // 2 + sway
        y = 310 + bob

        bg_rgba = bg.convert("RGBA")
        bg_rgba.alpha_composite(actor, (x, y))

        d = ImageDraw.Draw(bg_rgba)

        # talking mouth test overlay near lower face area
        # This is only a test mouth marker; later we replace with real mouth layer.
        mouth_open = abs(math.sin(t * math.pi * 18))
        mx = width // 2
        my = y + int(actor.height * 0.36)
        mw = int(58 + 12 * mouth_open)
        mh = int(8 + 28 * mouth_open)
        d.ellipse((mx - mw//2, my - mh//2, mx + mw//2, my + mh//2), fill=(80, 20, 25, 210))

        # subtitle
        text = "Chu nhan, NAMI dang thu nhan vat noi va chuyen dong."
        d.rectangle((28, 1120, width-28, 1198), fill=(0,0,0,130))
        d.text((48, 1148), text, fill=(255,255,255,255))
        d.text((28, 36), "NAMI V55 - ACTOR TALKING + MOTION TEST", fill=(230,230,230,255))

        bg_rgba.convert("RGB").save(frames_dir / f"frame_{i:04d}.png")

    mp4 = out_dir / "nami_actor_talking_motion_v55.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {
            "ok": False,
            "status": "NAMI_2D_ACTOR_TALKING_MOTION_V55_BLOCKED",
            "reason": "ffmpeg missing"
        }

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(mp4)
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=240)

    if run.returncode != 0:
        return {
            "ok": False,
            "status": "NAMI_2D_ACTOR_TALKING_MOTION_V55_BLOCKED",
            "reason": "ffmpeg failed",
            "stderr": run.stderr[-2000:]
        }

    report = {
        "ok": True,
        "status": "NAMI_2D_ACTOR_TALKING_MOTION_V55_PASS",
        "created_at": datetime.utcnow().isoformat(),
        "video": str(mp4),
        "actor": str(actor_path),
        "duration_seconds": seconds,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "features": [
            "real_downloaded_actor_image_used",
            "breathing_bob_motion",
            "walking_sway_motion",
            "mouth_open_close_talking_test",
            "subtitle_test"
        ],
        "note": "Đây là test nhân vật thật nói/chuyển động; miệng chỉ là marker thử nghiệm, chưa phải lip-sync đẹp cuối cùng.",
        "next_stage": "NAMI_2D_ACTOR_WALK_CYCLE_V56"
    }

    report_file = out_dir / "actor_talking_motion_report_v55.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**report, "report": str(report_file)}


@app.get("/download/actor-talking-motion-v55")
def download_actor_talking_motion_v55():
    from pathlib import Path
    from fastapi.responses import FileResponse, JSONResponse

    f = Path("worker_outputs/nami_2d_actor_talking_motion_v55/nami_actor_talking_motion_v55.mp4")
    if not f.exists():
        return JSONResponse({
            "ok": False,
            "status": "ACTOR_TALKING_MOTION_V55_FILE_NOT_FOUND",
            "message": "Run task nami_2d_actor_talking_motion_v55 first."
        }, status_code=404)

    return FileResponse(
        str(f),
        media_type="video/mp4",
        filename="nami_actor_talking_motion_v55.mp4"
    )


def nami_2d_rig_learning_v56():
    import json
    from pathlib import Path
    from datetime import datetime

    lesson_file = Path("nami_learning/2d_rig/nami_2d_rig_learning_v56.json")
    if not lesson_file.exists():
        return {
            "ok": False,
            "status": "NAMI_2D_RIG_LEARNING_V56_BLOCKED",
            "reason": "lesson file missing",
            "need": str(lesson_file)
        }

    lesson = json.loads(lesson_file.read_text(encoding="utf-8"))

    out_dir = Path("worker_outputs/nami_2d_rig_learning_v56")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "ok": True,
        "status": "NAMI_2D_RIG_LEARNING_V56_PASS",
        "created_at": datetime.utcnow().isoformat(),
        "learned_core_rule": lesson["core_rule"],
        "main_engine": lesson["software_strategy"]["main_engine"],
        "safe_first_tools": lesson["software_strategy"]["safe_first_tools"],
        "next_pipeline": lesson["next_pipeline"],
        "important_change": "NAMI will stop treating flat illustration images as film actors. It must classify, cut out, layer, rig, then animate.",
        "next_stage": "NAMI_2D_ASSET_CLASSIFIER_V57"
    }

    out_file = out_dir / "nami_2d_rig_learning_report_v56.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {**report, "report": str(out_file)}


# === NAMI_WORKER_LEARN_2D_FILM_V96 ===
def nami_worker_learn_2d_film_v96():
    import time, json, hashlib

    owner_command = "Nami đi học thêm kiến thức làm phim 2D đi"

    learning_pack = {
        "status": "NAMI_WORKER_LEARN_2D_FILM_V96_PASS",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "HF_WORKER_STRUCTURED_LEARNING_V96",
        "owner_command": owner_command,
        "topic": "2D animation film production for NAMI",
        "summary": "Worker đã tạo gói kiến thức làm phim 2D có cấu trúc cho NAMI. Đây là bước học ngoài điện thoại, tập trung vào pipeline, rig, keyframe, in-between, lip-sync, render và kiểm tra chất lượng.",
        "modules": [
            {
                "name": "2D film pipeline",
                "lessons": [
                    "Phim 2D phải chia thành script, sequence, scene và shot.",
                    "Không render nguyên phim dài một lần; phải render theo shot rồi ghép.",
                    "Mỗi shot cần input rõ: nhân vật, bối cảnh, hành động, camera, thoại, âm thanh, thời lượng."
                ]
            },
            {
                "name": "Character consistency",
                "lessons": [
                    "Muốn nhân vật nhất quán phải có character sheet, model sheet, màu cố định, trang phục cố định và asset ID.",
                    "Mỗi nhân vật cần thư mục riêng gồm reference, sprite/layer, voice profile, motion profile.",
                    "Không trộn nhiều ảnh không cùng nhân vật vào cùng một vai chính."
                ]
            },
            {
                "name": "Rig and layer split",
                "lessons": [
                    "PNG phẳng không đủ để cử động tay chân thật.",
                    "Actor 2D nên tách tối thiểu: đầu, thân, tay trên trái/phải, tay dưới trái/phải, bàn tay, đùi, cẳng chân, bàn chân.",
                    "Sau khi tách layer mới gắn pivot/xương để tạo walk, run, fight, idle, talk."
                ]
            },
            {
                "name": "Keyframe and in-between",
                "lessons": [
                    "Keyframe là pose chính thể hiện hành động.",
                    "In-between là các frame trung gian giúp chuyển động mượt.",
                    "Motion phải có timing, easing, anticipation, follow-through, overlap."
                ]
            },
            {
                "name": "Lip-sync",
                "lessons": [
                    "Lip-sync cần bộ miệng/viseme: neutral, A, E, O, U, M/B/P, F/V.",
                    "Thoại phải chia theo câu, thời gian và cảm xúc.",
                    "Nếu chưa có miệng tách layer thì chỉ làm talking motion giả, chưa đạt chất lượng phim."
                ]
            },
            {
                "name": "Render and QA",
                "lessons": [
                    "Render xong phải kiểm tra file thật: dung lượng, mở được, thời lượng, khung hình, âm thanh.",
                    "File vài chục byte là lỗi, không tính là video PASS.",
                    "Output hợp lệ phải lưu vào /sdcard/Download/NAMI_2D_TESTS để app tab Video tải về hiện ra."
                ]
            }
        ],
        "next_worker_tasks": [
            "V97 tạo actor 005 layer-split plan.",
            "V98 tạo rig requirement và motion bank cho actor 005.",
            "V99 render một test có tay/chân cử động rõ hơn.",
            "V100 gửi output mp4 về tab Video tải về."
        ]
    }

    learning_pack["content_hash"] = hashlib.sha256(
        json.dumps(learning_pack, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return learning_pack
# === END_NAMI_WORKER_LEARN_2D_FILM_V96 ===


# === NAMI_WORKER_ACTOR_005_LAYER_RIG_PLAN_V97 ===
def nami_worker_actor_005_layer_rig_plan_v97():
    import time, json, hashlib

    plan = {
        "status": "NAMI_WORKER_ACTOR_005_LAYER_RIG_PLAN_V97_PASS",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "HF_WORKER_ACTOR_RIG_PLANNING_V97",
        "actor": "actor_005",
        "goal": "Chuẩn bị cho actor 005 cử động tay chân thật hơn bằng layer split và rig plan.",
        "problem": [
            "Actor 005 hiện là PNG phẳng.",
            "PNG phẳng chỉ làm được motion illusion, chưa làm được walk/run/fight tự nhiên.",
            "Muốn tay chân cử động thật cần tách bộ phận và gắn pivot/xương."
        ],
        "required_layers": [
            "head",
            "neck",
            "torso",
            "upper_arm_left",
            "lower_arm_left",
            "hand_left",
            "upper_arm_right",
            "lower_arm_right",
            "hand_right",
            "thigh_left",
            "shin_left",
            "foot_left",
            "thigh_right",
            "shin_right",
            "foot_right",
            "hair_front",
            "hair_back",
            "eye_left",
            "eye_right",
            "mouth_set"
        ],
        "pivot_points": {
            "head": "neck joint",
            "upper_arm_left": "left shoulder",
            "lower_arm_left": "left elbow",
            "hand_left": "left wrist",
            "upper_arm_right": "right shoulder",
            "lower_arm_right": "right elbow",
            "hand_right": "right wrist",
            "thigh_left": "left hip",
            "shin_left": "left knee",
            "foot_left": "left ankle",
            "thigh_right": "right hip",
            "shin_right": "right knee",
            "foot_right": "right ankle"
        },
        "motion_tests": [
            {
                "name": "idle_breathing",
                "description": "Thân và đầu nhún nhẹ, tay dao động nhỏ."
            },
            {
                "name": "fear_step_back",
                "description": "Nhân vật lùi một bước, vai co lại, tay đưa lên phòng thủ."
            },
            {
                "name": "walk_cycle_basic",
                "description": "Hai chân luân phiên bước, tay đánh ngược pha với chân."
            },
            {
                "name": "run_cycle_basic",
                "description": "Chân bước dài hơn, thân nghiêng nhẹ, timing nhanh."
            },
            {
                "name": "zombie_reaction",
                "description": "Giật mình, lùi lại, xoay đầu nhìn zombie."
            }
        ],
        "qa_rules": [
            "Không tính PASS nếu chỉ kéo cả PNG qua lại.",
            "PASS tối thiểu phải thấy ít nhất 2 khớp tay hoặc 2 khớp chân xoay độc lập.",
            "File video phải lớn hơn 10KB và mở được.",
            "Nếu chưa tách layer thật thì phải ghi BLOCKED_BY_FLAT_PNG, không được báo đạt."
        ],
        "next_worker_tasks": [
            "V98: tạo local layer split manifest cho actor 005.",
            "V99: tạo puppet rig test dùng layer giả/cutout an toàn.",
            "V100: render motion test có tay/chân cử động rõ hơn.",
            "V101: gửi mp4 hợp lệ vào /sdcard/Download/NAMI_2D_TESTS để app tab Video tải về hiện ra."
        ]
    }

    plan["content_hash"] = hashlib.sha256(
        json.dumps(plan, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return plan
# === END_NAMI_WORKER_ACTOR_005_LAYER_RIG_PLAN_V97 ===


# NAMI AutoCap extension — preserves all existing routes.
from autocap_routes import register_autocap_routes
register_autocap_routes(app)
