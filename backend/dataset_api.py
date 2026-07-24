"""
Dataset / batch evaluation API for PhysicsLENS.
---------------------------------------------
A self-contained FastAPI router that adds batch evaluation over many videos
without touching the single-video flow in main.py. Two ways to load videos:

  • Browser upload  — POST /dataset/upload  (multiple files / a folder)
  • HuggingFace     — POST /dataset/hf_download  (streams progress; grabs every
                      video file — mp4/webm/mov/gif/… — from a dataset repo,
                      public or gated via token)

Both register each file under a short id. The frontend then runs any selected
pipeline on a video by id via POST /dataset/run (NDJSON stream, identical event
schema to /run). Because every pipeline runs against the *same* on-disk file,
the per-video track cache + evidence bus are shared across the batch for free.

Files live in a managed temp directory; clients reference them by id only, so
no arbitrary server path is ever exposed.
"""
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Extensions we accept as "videos" (animated GIFs included). Broad allowlist so a
# dropped folder of mixed media only feeds real clips to the decoder.
VIDEO_EXTS = {
    ".gif", ".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv",
    ".mpg", ".mpeg", ".ogv", ".wmv", ".flv", ".3gp", ".ts", ".mts", ".m2ts",
}

# Per-user directory: on a shared host the temp root is world-visible, so a
# fixed name like "physicslens_dataset" gets created by whoever starts a server
# first and everyone else hits PermissionError writing into it. Suffixing with
# a per-user id gives each user their own writable directory.
# os.getuid() is POSIX-only; on Windows fall back to the login name (hashed for
# a filesystem-safe suffix).
try:
    _USER_ID = str(os.getuid())                                   # POSIX
except AttributeError:                                            # Windows
    import getpass
    _USER_ID = hashlib.sha1(getpass.getuser().encode("utf-8", "replace")).hexdigest()[:12]

DATASET_DIR = Path(tempfile.gettempdir()) / f"physicslens_dataset_{_USER_ID}"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(DATASET_DIR, 0o700)          # best-effort; no-op / unsupported on Windows
except (OSError, NotImplementedError):
    pass

# id -> {"path": Path, "name": str}
_FILES: dict[str, dict] = {}


def _register(path: Path, name: str) -> dict:
    fid = uuid.uuid4().hex[:12]
    _FILES[fid] = {"path": Path(path), "name": name}
    return {"id": fid, "name": name}


def _nd(obj: dict) -> str:
    return json.dumps(obj) + "\n"


def build_dataset_router(pipelines: dict) -> APIRouter:
    """Build the /dataset router, closing over the pipeline registry."""
    router = APIRouter(prefix="/dataset", tags=["dataset"])

    # ── Browser upload (files or a whole folder) ──────────────────────────────
    @router.post("/upload")
    async def upload(files: List[UploadFile] = File(...)):
        out, skipped = [], 0
        for up in files:
            ext = Path(up.filename or "").suffix.lower()
            if ext not in VIDEO_EXTS:
                skipped += 1
                continue
            dest = DATASET_DIR / f"{uuid.uuid4().hex}{ext}"
            with open(dest, "wb") as fh:
                shutil.copyfileobj(up.file, fh)
            out.append(_register(dest, Path(up.filename).name))
        return {"files": out, "skipped": skipped}

    # ── HuggingFace download (streaming progress) ─────────────────────────────
    @router.post("/hf_download")
    async def hf_download(
        repo_id:   str           = Form(...),
        token:     Optional[str] = Form(None),
        subfolder: Optional[str] = Form(None),
        max_files: int           = Form(0),
    ):
        async def stream():
            try:
                from huggingface_hub import HfApi, hf_hub_download
            except Exception as exc:
                yield _nd({"type": "error",
                           "text": f"huggingface_hub not installed ({exc}). "
                                   f"Run: pip install huggingface_hub"})
                return

            tok = (token or "").strip() or None
            repo = repo_id.strip()
            yield _nd({"type": "log", "level": "info",
                       "text": f"Listing files in dataset “{repo}”…"})
            try:
                api = HfApi()
                all_files = api.list_repo_files(repo, repo_type="dataset", token=tok)
            except Exception as exc:
                yield _nd({"type": "error",
                           "text": f"Could not list “{repo}”: {exc}"})
                return

            vids = [f for f in all_files if Path(f).suffix.lower() in VIDEO_EXTS]
            if subfolder:
                sf = subfolder.strip().strip("/")
                vids = [f for f in vids if f.startswith(sf + "/")]
            vids.sort()
            if max_files and max_files > 0:
                vids = vids[:max_files]

            if not vids:
                yield _nd({"type": "error",
                           "text": "No video files found in that dataset"
                                   + (f" under “{subfolder}”." if subfolder else ".")})
                return

            total = len(vids)
            yield _nd({"type": "log", "level": "info",
                       "text": f"Found {total} video file(s). Downloading…"})

            for i, rel in enumerate(vids):
                try:
                    local = hf_hub_download(repo, rel, repo_type="dataset", token=tok)
                    rec = _register(Path(local), Path(rel).name)
                    yield _nd({"type": "file", "id": rec["id"], "name": rec["name"],
                               "rel": rel, "index": i + 1, "total": total})
                except Exception as exc:
                    yield _nd({"type": "log", "level": "warn",
                               "text": f"Skipped {rel}: {exc}"})
                yield _nd({"type": "progress", "done": i + 1, "total": total})

            yield _nd({"type": "done", "total": total})

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    # ── Serve a registered file for in-browser preview ────────────────────────
    @router.get("/file/{file_id}")
    def serve_file(file_id: str):
        rec = _FILES.get(file_id)
        if not rec or not rec["path"].exists():
            return JSONResponse({"error": "Unknown file id"}, status_code=404)
        return FileResponse(str(rec["path"]), filename=rec["name"])

    # ── Run a pipeline on a registered file (NDJSON stream, like /run) ─────────
    @router.post("/run")
    async def run_by_id(
        pipeline_id: str           = Form(...),
        file_id:     str           = Form(...),
        prompt:      Optional[str] = Form(None),
        settings:    Optional[str] = Form(None),
    ):
        if pipeline_id not in pipelines:
            return JSONResponse({"error": f"Unknown pipeline: {pipeline_id}"},
                                status_code=400)
        rec = _FILES.get(file_id)
        if not rec or not rec["path"].exists():
            return JSONResponse({"error": "Unknown file id"}, status_code=404)

        p = pipelines[pipeline_id]
        path = str(rec["path"])

        kwargs: dict = {}
        if p.get("requires_prompt"):
            kwargs["prompt"] = prompt or ""
        if p.get("settings"):
            kwargs["settings"] = settings

        async def event_stream():
            try:
                from tools.costs import instrument
                if p.get("requires_pair"):
                    gen = p["run"](path, path, **kwargs)   # self-pair fallback
                else:
                    gen = p["run"](path, **kwargs)
                gen = instrument(gen, badge=p.get("badge", "—"))
                async for event in gen:
                    yield json.dumps(event) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "error", "text": str(exc)}) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    # ── Temporal segment: trim a registered video to [t0, t1) as a new file ────
    @router.post("/segment")
    def segment(file_id: str = Form(...), t0: float = Form(...), t1: float = Form(...)):
        rec = _FILES.get(file_id)
        if not rec or not rec["path"].exists():
            return JSONResponse({"error": "Unknown file id"}, status_code=404)
        if t1 <= t0:
            return JSONResponse({"error": "t1 must be greater than t0"}, status_code=400)

        from tools.video import load_frames, save_frames_as_video

        frames, fps = load_frames(str(rec["path"]))
        n = len(frames)
        if n < 2 or fps <= 0:
            return JSONResponse({"error": "Source video too short/unreadable to segment"},
                                status_code=400)
        duration = n / fps
        i0 = max(0, min(n - 1, int(round(t0 * fps))))
        i1 = max(i0 + 1, min(n, int(round(t1 * fps))))
        seg = frames[i0:i1]
        if len(seg) < 2:
            return JSONResponse(
                {"error": f"Segment [{t0:g}, {t1:g})s is empty/too short after "
                          f"clamping to the video's 0–{duration:.2f}s range."},
                status_code=400)

        dest = DATASET_DIR / f"{uuid.uuid4().hex}.mp4"
        save_frames_as_video(seg, fps, str(dest))
        stem = Path(rec["name"]).stem
        name = f"{stem}_seg{i0 / fps:.2f}-{i1 / fps:.2f}s.mp4"
        out = _register(dest, name)
        return {**out, "parent_id": file_id, "t0": round(i0 / fps, 3),
                "t1": round(i1 / fps, 3), "fps": fps, "n_frames": len(seg)}

    # ── Clear the in-memory registry (files are temp; left for the OS) ─────────
    @router.post("/clear")
    def clear():
        n = len(_FILES)
        _FILES.clear()
        return {"cleared": n}

    return router
