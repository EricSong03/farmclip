"""farmclip CLI: full pipeline — clip -> calibrate -> ball -> players -> scene.json.

Usage: farmclip VIDEO --start 60 --end 720 --out out/mikasa
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import video
from .scene import write_scene

VBALLNET = Path(__file__).parent.parent / "vendor" / "fast-vb-tracking"
# dual-model union: Grid catches what V1c misses (4px median agreement when
# both fire; union measured +10pts in-play coverage over either alone)
VBALLNET_MODELS = [
    VBALLNET / "models" / "VballNetGridV1b_seq15_grayscale_20260427_194144.onnx",
    VBALLNET / "models" / "VballNetV1c_seq9_grayscale_best.onnx",
]


def _calibrate_ai(clip: Path, out: Path, w: int, h: int, n_frames: int,
                  model: Path, frames_out: list | None = None) -> dict | None:
    """AI keypoint path: per-keypoint median over ~10 frames -> click-grade solve.

    Same math as the manual labeling path (solve_web): pose anchored on floor
    points, net height MEASURED from net/antenna detections, multi-start.
    Rejects hallucinated detections: worst floor point dropped iteratively
    while floor err > 15px, and the whole calib refused if it stays > 25px —
    better no calib (hough/manual fallback) than a confidently wrong one.
    """
    from .kp_detect import detect_keypoints
    from .calibrate import KEYPOINTS, draw_overlay, solve_web

    hits, ref = {}, None
    for i, t, frame in video.read_frames(str(clip), step=max(1, int(n_frames // 10)),
                                         seek=True):
        if ref is None:
            ref = (i, frame)
        if frames_out is not None:
            frames_out.append(frame)
        for name, uv in detect_keypoints(frame, str(model)).items():
            hits.setdefault(name, []).append(uv)
    # median across frames kills per-frame jitter/occlusion; >=3 sightings only
    pts = {n: tuple(np.median(v, axis=0)) for n, v in hits.items() if len(v) >= 3}
    if len(pts) < 5:
        print(f"ai_kp: only {len(pts)} stable keypoints (<5)")
        return None
    try:
        calib, net_h = solve_web(pts, w, h)
        while calib["err"] > 15:
            floor = [n for n in calib.get("per_point_err", {}) if KEYPOINTS[n][1] == 0]
            if len(floor) <= 6:
                break
            worst = max(floor, key=calib["per_point_err"].get)
            pts.pop(worst, None)
            print(f"ai_kp: dropping outlier {worst} "
                  f"({calib['per_point_err'][worst]:.0f}px), re-solving")
            calib, net_h = solve_web(pts, w, h)
    except Exception as e:
        print(f"ai_kp: solve failed ({e})")
        return None
    if calib["err"] > 12:  # good anchors land <10; hallucinations fit ~20-25 "consistently"
        print(f"ai_kp: floor err {calib['err']:.1f}px > 12 — rejecting AI anchor")
        # still worth returning: too loose to ship, good enough to seed the
        # line solve, which re-fits the pose against dense line evidence
        return dict(calib, rejected=True)
    # Reprojection error alone cannot catch a self-consistent WRONG pose: v6b
    # produced a 7.0px anchor on a head-on clip whose net projected 262px off
    # the real net band. Check the net independently — it is the only
    # high-contrast structure off the floor, so a wrong pose cannot place it.
    # Scored on a MEDIAN frame, not a raw one, and that is not a detail:
    # players and equipment standing near a wrongly-projected net supply
    # spurious line-like evidence. Measured on the same calib — raw frame said
    # the net matched at 5.7px, the median frame said 262.5px. The median
    # erases everything that moves and leaves an unoccluded court.
    from .calib_score import net_ok, score
    try:
        judge_frame = video.median_frame(clip)
    except Exception:
        judge_frame = ref[1]
    ok, why = net_ok(score(judge_frame, calib, net_h=calib.get("net_h_est")))
    if ok is False:
        print(f"ai_kp: {calib['err']:.1f}px floor fit but {why} — rejecting AI anchor")
        return dict(calib, rejected=True)
    print(f"ai_kp: net check — {why}")
    calib["method"] = "ai_kp"
    calib["ref_frame"] = ref[0]
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    dbg = out / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / "calib_overlay.jpg"),
                draw_overlay(ref[1], calib, pts, net_h=calib.get("net_h_est")))
    print(f"calibrated via ai_kp: {len(pts)} keypoints, err {calib['err']:.1f}px "
          f"-> {dbg / 'calib_overlay.jpg'}")
    return calib


def _calibrate_lineseg(out: Path, frames: list, init: dict,
                       model: Path, max_err: float = 6.0) -> dict | None:
    """Line-segmentation refine of an existing calib (init from keypoints/hough).

    The UNet segments the 8 named court lines; the pose is re-fit against every
    line pixel instead of a handful of corners. Init only has to be roughly
    right — this has no global search, but dense line evidence pulls a sloppy
    keypoint pose (or a rejected one) onto the actual paint.
    """
    from .lineseg import detect_lines, draw_mask, solve_lines
    from .calibrate import draw_overlay

    pix = detect_lines(frames, str(model))
    if len(pix) < 3:  # 2 lines can't fix both court directions
        print(f"lineseg: only {len(pix)} line classes segmented (<3)")
        return None
    try:
        calib, med = solve_lines(pix, init, net_h=init.get("net_h_est") or 2.43)
    except Exception as e:
        print(f"lineseg: solve failed ({e})")
        return None
    if med > max_err:
        print(f"lineseg: {med:.1f}px median line error > {max_err} — keeping init")
        return None
    calib.pop("rejected", None)
    if "net_h_est" not in calib and init.get("net_h_est"):
        calib["net_h_est"] = init["net_h_est"]  # lineseg couldn't see it, keypoints could
    calib["ref_frame"] = init.get("ref_frame")
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    dbg = out / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / "calib_overlay.jpg"),
                draw_overlay(frames[0], calib, net_h=calib.get("net_h_est")))
    cv2.imwrite(str(dbg / "lineseg_mask.jpg"), draw_mask(frames[0], pix))
    print(f"calibrated via lineseg: {len(pix)} lines, {med:.1f}px median "
          f"(from {init.get('method', 'init')} @ {init['err']:.1f}px) "
          f"-> {dbg / 'calib_overlay.jpg'}")
    return calib


def calibrate(clip: Path, out: Path):
    """Auto-calibration: AI keypoints -> line-segmentation refine, else Hough."""
    from .calibrate import draw_overlay
    from .hypothesis import solve_frame

    info_v = video.video_info(clip)
    w, h = info_v["width"], info_v["height"]
    kp_model = Path("finetune_out/yolo11s-court.onnx")
    seg_model = Path("finetune_out/lineseg/lineseg.onnx")
    calib, frames = None, []
    if kp_model.exists():
        calib = _calibrate_ai(clip, out, w, h, info_v["frames"], kp_model,
                              frames_out=frames)
    if calib is not None and seg_model.exists():
        calib = _calibrate_lineseg(out, frames, calib, seg_model) or calib
    if calib is not None and not calib.get("rejected"):
        return calib
    print("ai calibration failed, falling back to hough search")
    step = max(1, int(info_v["frames"] // 40))
    best = None
    for i, t, frame in video.read_frames(str(clip), step=step, seek=True):
        calib, info = solve_frame(frame, w, h)
        if calib is None:
            continue
        score = (info["n_inliers"], -calib["err"])
        if best is None or score > best[0]:
            best = (score, calib, i, frame)
            print(f"  calib candidate @frame {i}: err {calib['err']:.1f}, "
                  f"inliers {info['n_inliers']}")
    if best is None:
        sys.exit("calibration failed: no frame produced a valid hypothesis")
    _, calib, ref_frame, frame = best
    calib["method"] = "hough"
    calib["ref_frame"] = ref_frame
    if seg_model.exists():  # dense lines beat 4 hough segments when they agree
        calib = _calibrate_lineseg(out, [frame], calib, seg_model) or calib
        if calib["method"] == "lineseg":
            return calib
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    dbg = out / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / "calib_overlay.jpg"), draw_overlay(frame, calib))
    print(f"calibrated via hough on frame {ref_frame} (err {calib['err']:.1f}px) "
          f"-> {dbg / 'calib_overlay.jpg'}")
    return calib


def run_ball(clip: Path, out: Path, calib: dict, fps: float):
    """Dual VballNet union 2D -> ballistic 3D. Returns {frame: [x,y,z]}."""
    import pandas as pd
    from .ball3d import lift_with_streaks

    dfs = []
    for i, model in enumerate(VBALLNET_MODELS):
        mdir = out / f"ball{i}"
        csvs = list(mdir.glob("*/ball.csv"))
        if not csvs:
            subprocess.run(
                [sys.executable, "src/inference_onnx_seq_gray_v2.py",
                 "--video_path", str(clip.resolve()),
                 "--model_path", str(model.resolve()),
                 "--output_dir", str(mdir.resolve()), "--only_csv"],
                cwd=VBALLNET, check=True)
            csvs = list(mdir.glob("*/ball.csv"))
        dfs.append(pd.read_csv(csvs[0]))
    # physics-first (docs/specs/ball-physics-first.md): consensus anchors
    # only, velocity-break segmentation, no detection-side interpolation
    from .ball3d import weighted_track, reject_outliers, measure_net_bands
    measure_net_bands(clip, calib, fps)
    cols = ["Frame", "X", "Y", "Visibility"]
    track = weighted_track(dfs[0][cols].to_numpy(float),
                           dfs[1][cols].to_numpy(float))
    track = reject_outliers(track)
    ball3d, n_seg, n_ok, _ = lift_with_streaks(track, calib, fps, clip)
    print(f"ball: {int((track[:, 3] > 0).mean() * 100)}% consensus anchors, "
          f"{n_ok}/{n_seg} segments fitted, {len(ball3d)} 3D frames")
    return ball3d


def run_players(clip: Path, calib: dict, model_name="yolo11s.pt", imgsz=1280,
                conf=0.2, per_side=6, step=1, coast_s=1.5, fps=30.0):
    """YOLO persons -> tracks -> ground positions. Returns {frame: [players]}.
    step>1 subsamples frames (capsules don't need 60Hz); coast_s bridges misses."""
    from ultralytics import YOLO
    from .players import Tracker, to_scene_players

    model = YOLO(model_name)
    tracker = Tracker(max_miss=int(coast_s * fps / step))
    cap = cv2.VideoCapture(str(clip))
    per_frame = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        r = model.predict(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        tracked = tracker.update([b.xyxy[0].tolist() for b in r.boxes])
        hits = {tid: t.get("hits", 0) for tid, t in tracker.tracks.items()}
        players = to_scene_players(tracked, calib, per_side=per_side, hits=hits)
        if players:
            per_frame[i] = players
        i += 1
    cap.release()
    counts = [len(v) for v in per_frame.values()]
    print(f"players: {len(per_frame)}/{i} frames, "
          f"avg on-court {sum(counts) / max(len(counts), 1):.1f}")
    return per_frame


def main(argv=None):
    ap = argparse.ArgumentParser(prog="farmclip")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=60.0)
    ap.add_argument("--out", default="out")
    _ft = "finetune_out/yolo11s-vb.pt"  # H200 fine-tune, see docs/plans/icrn-finetune-results.md
    ap.add_argument("--player-model",
                    default=_ft if Path(_ft).exists() else "yolo11s.pt")
    ap.add_argument("--player-imgsz", type=int, default=1280)
    ap.add_argument("--per-side", type=int, default=6)
    ap.add_argument("--player-step", type=int, default=1)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "clip.mp4"
    if not clip.exists():
        print(f"clipping {args.start}-{args.end}s -> {clip}")
        video.clip_video(args.video, clip, args.start, args.end)
    info = video.video_info(clip)
    fps = info["fps"]
    print(f"clip: {info['width']}x{info['height']} @ {fps:.2f}fps, {info['frames']} frames")

    calib_path = out / "calib.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else calibrate(clip, out)

    ball3d = run_ball(clip, out, calib, fps)
    players = run_players(clip, calib, model_name=args.player_model,
                          imgsz=args.player_imgsz, per_side=args.per_side,
                          step=args.player_step, fps=fps)

    frames = []
    for i in sorted(set(ball3d) | set(players)):
        frames.append({"t": i / fps, "ball": ball3d.get(i), "players": players.get(i, [])})
    path = write_scene(out / "scene.json", frames,
                       net_height=round(calib.get("net_h", 2.43), 2))
    print(f"scene: {path} ({len(frames)} frames, "
          f"{sum(1 for f in frames if f.get('ball'))} ball, "
          f"{sum(1 for f in frames if f.get('players'))} players)")


if __name__ == "__main__":
    main()
