# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi", "uvicorn", "python-multipart"]
# ///
"""farmclip local app.

    uv run farmclip.py serve      # http://localhost:8000, upload in the browser
    uv run farmclip.py <video>    # headless: same pipeline, prints scene.json path
"""
import shutil
import sys
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent
JOBS_DIR = ROOT / "out" / "jobs"
JOBS: dict[str, dict] = {}  # ponytail: in-memory + out/ folder; a DB when hosting matters

STAGES = ["calibrate", "ball", "players"]

app = FastAPI()


def run_pipeline(job_id: str, video: Path) -> None:
    job = JOBS[job_id]
    try:
        for stage in STAGES:
            job["stage"] = stage
            time.sleep(2)  # ponytail: stub — real stages replace this loop body (see docs/plans/roadmap.md)
        shutil.copy(ROOT / "examples" / "sample-scene.json", video.parent / "scene.json")
        job["stage"] = "done"
        job["done"] = True
    except Exception as e:
        job["error"] = str(e)
        job["done"] = True


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.post("/api/upload")
def upload(file: UploadFile, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:8]
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    dest = d / ("input" + Path(file.filename or "video.mp4").suffix)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    JOBS[job_id] = {"stage": "queued", "done": False, "error": None}
    bg.add_task(run_pipeline, job_id, dest)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}/scene.json")
def job_scene(job_id: str):
    p = JOBS_DIR / job_id / "scene.json"
    if not p.exists():
        raise HTTPException(404, "not finished")
    return FileResponse(p)


def main():
    args = sys.argv[1:]
    if args and args[0] == "serve":
        import uvicorn
        uvicorn.run(app, port=8000)
    elif args:
        video = Path(args[0])
        if not video.exists():
            sys.exit(f"no such file: {video}")
        job_id = "cli"
        d = JOBS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        dest = d / video.name
        shutil.copy(video, dest)
        JOBS[job_id] = {"stage": "queued", "done": False, "error": None}
        run_pipeline(job_id, dest)
        print(d / "scene.json")
    else:
        print(__doc__.strip())


if __name__ == "__main__":
    main()
