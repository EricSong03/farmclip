"""Build out/debug/index.html: every debug image/video, newest first.

Usage: python scripts/debug_index.py   (rerun any time to refresh)
"""
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIRS = [ROOT / "out" / "debug", ROOT / "out" / "mikasa" / "debug"]
DEST = ROOT / "out" / "debug" / "index.html"

items = []
for d in DIRS:
    if not d.exists():
        continue
    for p in d.iterdir():
        if p.suffix.lower() in (".jpg", ".png", ".mp4") and p.name != "index.html":
            items.append(p)
items.sort(key=lambda p: p.stat().st_mtime, reverse=True)

rows = []
for p in items:
    rel = Path("..") / p.relative_to(ROOT / "out") if "mikasa" in str(p) \
        else p.relative_to(ROOT / "out" / "debug")
    rel = str(rel).replace("\\", "/")
    ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d %H:%M")
    clip = "mikasa" if "mikasa" in str(p) else "menlo"
    body = (f'<video src="{rel}" controls preload="metadata"></video>'
            if p.suffix == ".mp4" else f'<a href="{rel}"><img src="{rel}" loading="lazy"></a>')
    rows.append(f'<figure>{body}<figcaption><b>{p.name}</b> — {clip} — {ts}'
                f'</figcaption></figure>')

DEST.write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>farmclip debug — newest first</title>
<style>
 body {{ background:#111; color:#ddd; font:14px system-ui; margin:16px; }}
 figure {{ margin:0 0 28px 0; }}
 img, video {{ max-width:100%; border:1px solid #333; }}
 figcaption {{ padding:6px 2px; color:#aaa; }} b {{ color:#ffd54f; }}
</style></head><body>
<h2>debug artifacts — newest first ({len(items)})</h2>
{''.join(rows)}
</body></html>""", encoding="utf-8")
print(f"{DEST}: {len(items)} items")
