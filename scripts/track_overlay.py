"""Per-frame camera track + wireframe overlay video.

Usage: python scripts/track_overlay.py <outdir> <video> [step]
Anchor: <outdir>/calib.json at frame <outdir>/ref_idx.txt (default 0).
If calib.json is missing but finetune_out/yolo11s-court.onnx exists, the
anchor is solved from AI keypoints on the ref frame.
Writes <outdir>/camera_track.json + <outdir>/debug/overlay_track.mp4 (H.264).
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.calibrate import draw_overlay
from farmclip.track import track_camera

outdir, video = Path(sys.argv[1]), sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ref_path = outdir / "ref_idx.txt"
ref_idx = int(ref_path.read_text().strip()) if ref_path.exists() else 0

cpath = outdir / "calib.json"
if cpath.exists():
    calib0 = json.loads(cpath.read_text())
else:
    onnx = Path("finetune_out/yolo11s-court.onnx")
    if not onnx.exists():
        sys.exit(f"{cpath} missing and no {onnx} to auto-anchor")
    from farmclip.calibrate import solve_auto
    from farmclip.kp_detect import detect_keypoints
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read frame {ref_idx} of {video}")
    calib0, _ = solve_auto(detect_keypoints(frame, str(onnx)),
                           frame.shape[1], frame.shape[0])
    print(f"auto-anchored via ai_kp on frame {ref_idx}: err {calib0['err']:.1f}px")

track = track_camera(video, calib0, ref_idx=ref_idx, step=step)

out = {str(i): {"rvec": [round(v, 6) for v in c["rvec"]],
                "tvec": [round(v, 6) for v in c["tvec"]],
                "f": round(c["f"], 6), "err": round(c["err"], 6)}
       for i, c in sorted(track.items())}
(outdir / "camera_track.json").write_text(json.dumps(out))
print(f"wrote {outdir / 'camera_track.json'} ({len(out)} frames)")

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
tmp = outdir / "debug" / "_track_tmp.mp4"
dest = outdir / "debug" / "overlay_track.mp4"
tmp.parent.mkdir(parents=True, exist_ok=True)
vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
f = 0
last = track[ref_idx]
while True:
    ok, img = cap.read()
    if not ok:
        break
    c = track.get(f, last)  # step>1: hold nearest earlier tracked pose
    last = c
    img = draw_overlay(img, c)
    cv2.putText(img, f"f{f} err {c['err']:.1f}px n{c.get('n', 0)}",
                (10, h - 14), 0, 0.6, (0, 255, 255), 2)
    vw.write(img)
    f += 1
cap.release()
vw.release()

ff = Path(sys.prefix) / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries"
ffexe = next(ff.glob("ffmpeg-*"), None)
if ffexe:
    subprocess.run([str(ffexe), "-y", "-v", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    "-an", str(dest)], check=True)
    tmp.unlink()
else:
    tmp.rename(dest)  # ponytail: mp4v fallback, plays in VLC not browsers
print(f"wrote {dest} ({f} frames)")
