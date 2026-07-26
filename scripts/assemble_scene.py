"""Merge out/ball3d.json + out/players.json -> out/scene.json (spec v1)."""
import json

from farmclip.scene import write_scene

FPS = 30.0
ball = {int(k): v for k, v in json.load(open("out/ball3d.json")).items()}
players = {int(k): v for k, v in json.load(open("out/players.json")).items()}
calib = json.load(open("out/calib.json"))

frames = []
for i in sorted(set(ball) | set(players)):
    frames.append({
        "t": i / FPS,
        "ball": ball.get(i),
        "players": players.get(i, []),
    })
path = write_scene("out/scene.json", frames, net_height=round(calib.get("net_h", 2.43), 2))
n_ball = sum(1 for f in frames if f.get("ball"))
print(f"{path}: {len(frames)} frames, {n_ball} with ball, "
      f"{sum(1 for f in frames if f.get('players'))} with players")
