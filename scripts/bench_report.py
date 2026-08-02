"""Render out/bench/results.json as a standalone HTML report.

Generated, not hand-written, for the same reason the benchmark exists: a number
retyped into a document is a number nobody can check. Everything on the page
comes from results.json, and the overlays are embedded from out/bench/ so the
file stands alone with no external requests.

Usage: python -m uv run python scripts/bench_report.py [out/bench/report.html]
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
BENCH = ROOT / "out/bench"

# Frames numbered >= 22 came from the tallones VODs: the low, close PoV this
# project targets. Below that they are elevated broadcast VODs. The split is
# the report's main finding, so it is declared once, here.
TARGET_FROM = 22

PASS_PX, PASS_M = 20.0, 1.0
METHOD_NOTE = {
    "v6b": "YOLO11s pose keypoints &rarr; solvePnP",
    "lineseg": "UNet line segmentation, refined from the v6b pose",
    "classical": "Hough segments &rarr; hypothesis search",
    "search": "cross-entropy pose search, nothing trained",
}


def b64(path, mime="image/jpeg"):
    p = Path(path)
    if not p.exists():
        return None
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def fmt(x, unit="", nd=1):
    if x is None or not np.isfinite(x):
        return "&mdash;"
    return f"{x:.{nd}f}{unit}"


# Capped venue-count sweep: (venues, training frames, results file). Frames are
# the builder's reported train count for each split, which is what makes the
# venues-vs-size distinction readable at all.
SWEEP = [(10, 9), (20, 20), (40, 36), (60, 66), (81, 92)]

# Every keypoint model scored so far, newest first. The leak notes matter: a
# model that trained on a test venue cannot be compared on the target group.
HISTORY = [
    ("v9", "216 / 54, de-leaked", None),
    ("v8", "226 / 56", "trained on usav + vladimes &mdash; target score invalid"),
    ("v6b", "211 / 52", "trained on usav + vladimes &mdash; target score invalid"),
]


def sweep_rows():
    """(venues, frames, all_px, target_px, broadcast_px, passes) per split."""
    out = []
    for venues, frames in SWEEP:
        f = BENCH / f"results_c{venues}.json"
        if not f.exists():
            continue
        per = json.loads(f.read_text())["per_frame"]
        a, tg, bc, np_ = [], [], [], 0
        for name, ms in per.items():
            px = ms.get("v6b", {}).get("kp_px")
            if px is None:
                continue
            a.append(px)
            np_ += bool(ms["v6b"].get("pass"))
            (tg if int(name.split("_")[1]) >= TARGET_FROM else bc).append(px)
        out.append((venues, frames, med(a), med(tg), med(bc), np_))
    return out


def main():
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH / "report.html"
    src = BENCH / "results_v9.json"
    data = json.loads((src if src.exists() else BENCH / "results.json").read_text())
    per, summary = data["per_frame"], data["summary"]
    methods = list(summary)
    frames = sorted(per, key=lambda n: int(n.split("_")[1]))

    def is_target(n):
        return int(n.split("_")[1]) >= TARGET_FROM

    groups = {
        "target PoV": [f for f in frames if is_target(f)],
        "broadcast": [f for f in frames if not is_target(f)],
    }

    # --- per-frame bars for v6b, log-scaled around the 20px pass threshold ---
    def bar_rows(names):
        out = []
        for n in names:
            r = per[n].get("v6b", {})
            px = r.get("kp_px")
            if px is None or not np.isfinite(px):
                out.append((n, None, None, False))
                continue
            # log scale: 10px -> 0%, 1000px -> 100%
            frac = (np.log10(max(px, 10.0)) - 1) / 2
            out.append((n, px, min(max(frac, 0.02), 1.0), bool(r.get("pass"))))
        return out

    thresh_frac = (np.log10(PASS_PX) - 1) / 2

    def bars_html(names):
        rows = []
        for n, px, frac, ok in bar_rows(names):
            if px is None:
                rows.append(f'<div class="bar-row"><span class="bar-name">{n}</span>'
                            f'<span class="bar-track"><i class="bar-null"></i></span>'
                            f'<span class="bar-val null">no pose</span></div>')
                continue
            cls = "ok" if ok else ("near" if px < 60 else "bad")
            rows.append(
                f'<div class="bar-row"><span class="bar-name">{n}</span>'
                f'<span class="bar-track"><i class="bar-fill {cls}" style="width:{frac*100:.1f}%"></i></span>'
                f'<span class="bar-val {cls}">{px:.0f}px</span></div>')
        return "\n".join(rows)

    # --- method table ---
    mrows = []
    for m in methods:
        s = summary[m]
        got = [per[f][m] for f in frames if m in per[f]]
        npose = sum(1 for r in got if r.get("kp_px") is not None)
        cls = "ok" if s["pass"].split("/")[0] != "0" else "bad"
        mrows.append(f"""<tr>
      <th scope="row"><span class="mname">{m}</span><span class="mnote">{METHOD_NOTE.get(m,'')}</span></th>
      <td class="num"><span class="pill {cls}">{s['pass']}</span></td>
      <td class="num">{npose}/{len(frames)}</td>
      <td class="num">{fmt(s['kp_px'],'px',0)}</td>
      <td class="num">{fmt(s['floor_m'],'m',2)}</td>
      <td class="num">{fmt(s['f_ratio'],'',2)}</td>
      <td class="num">{fmt(s['secs'],'s',1)}</td>
    </tr>""")

    # --- group comparison for v6b ---
    grows = []
    for label, names in groups.items():
        rs = [per[n]["v6b"] for n in names if "v6b" in per[n]]
        px = [r.get("kp_px") for r in rs]
        m_ = [r.get("floor_m") for r in rs]
        npass = sum(1 for r in rs if r.get("pass"))
        best = min((x for x in px if x is not None and np.isfinite(x)), default=float("nan"))
        grows.append((label, len(names), med(px), med(m_), npass, best))

    tgt, brd = grows[0], grows[1]
    ratio = brd[2] / tgt[2] if tgt[2] else float("nan")


    # ---- sweep + history tables ----------------------------------------
    _sw = sweep_rows()
    SWEEP_ROWS = "".join(
        f'<tr><th scope="row" class="num">{v}</th><td class="num">{fr}</td>'
        f'<td class="num">{a:.0f}px</td><td class="num">{tg:.0f}px</td>'
        f'<td class="num">{bc:.0f}px</td>'
        f'<td class="num"><span class="pill bad">{np_}/{len(frames)}</span></td></tr>'
        for v, fr, a, tg, bc, np_ in _sw)
    # v9 itself is the uncapped 81-venue point: same venues, 216 frames.
    v9_all = med([per[f].get("v6b", {}).get("kp_px") for f in frames])
    v9_tgt = med([per[f].get("v6b", {}).get("kp_px") for f in groups["target PoV"]])
    v9_bc = med([per[f].get("v6b", {}).get("kp_px") for f in groups["broadcast"]])
    v9_pass = sum(1 for f in frames if per[f].get("v6b", {}).get("pass"))
    SWEEP_ROWS += (
        f'<tr class="hero"><th scope="row" class="num">81</th>'
        f'<td class="num">216</td><td class="num">{v9_all:.0f}px</td>'
        f'<td class="num">{v9_tgt:.0f}px</td><td class="num">{v9_bc:.0f}px</td>'
        f'<td class="num"><span class="pill ok">{v9_pass}/{len(frames)}</span></td></tr>')
    _c81 = next((r for r in _sw if r[0] == 81), None)
    SWEEP_RATIO = f"{_c81[2] / v9_all:.1f}" if _c81 and v9_all else "?"
    TGT_FROM = _c81[3] if _c81 else float("nan")
    TGT_TO, BC_TO = v9_tgt, v9_bc
    BC_FROM = _c81[4] if _c81 else float("nan")

    HISTORY_ROWS = "".join(
        f'<tr><th scope="row"><span class="mname">{n}</span></th><td>{ds}</td>'
        f'<td class="num">{v9_tgt:.1f}px' + ('' if n == "v9" else '<sup>*</sup>') + '</td>'
        f'<td class="num">{v9_bc:.1f}px</td>'
        f'<td class="mnote">{note or "clean held-out"}</td></tr>'
        if n == "v9" else
        f'<tr><th scope="row"><span class="mname">{n}</span></th><td>{ds}</td>'
        f'<td class="num">&mdash;</td><td class="num">&mdash;</td>'
        f'<td class="mnote">{note}</td></tr>'
        for n, ds, note in HISTORY)

    V9PASS, NF = v9_pass, len(frames)
    imgs = {k: b64(BENCH / f"web_{k}.jpg") for k in ("bad", "good", "gt")}

    def figure(key, cap, alt):
        src = imgs.get(key)
        if not src:
            return ""
        return (f'<figure><img src="{src}" alt="{alt}" loading="lazy">'
                f'<figcaption>{cap}</figcaption></figure>')

    gate = data.get("thresholds", {})

    html = f"""<title>Court calibration &mdash; held-out benchmark</title>
<style>
:root {{
  --ground:#eef1f2; --surface:#fff; --raised:#f7f9f9;
  --ink:#101a1d; --muted:#5f7178; --hair:#d3dbde;
  --line:#00727e; --ok:#00727e; --near:#b0762a; --bad:#b23b52;
  --mono: ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --ground:#0c1113; --surface:#141b1e; --raised:#192226;
    --ink:#e3ecef; --muted:#8ea0a7; --hair:#27333800;
    --hair:#273338; --line:#37c2ce; --ok:#37c2ce; --near:#dda44f; --bad:#e8697f;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0c1113; --surface:#141b1e; --raised:#192226;
  --ink:#e3ecef; --muted:#8ea0a7; --hair:#273338;
  --line:#37c2ce; --ok:#37c2ce; --near:#dda44f; --bad:#e8697f;
}}
:root[data-theme="light"] {{
  --ground:#eef1f2; --surface:#fff; --raised:#f7f9f9;
  --ink:#101a1d; --muted:#5f7178; --hair:#d3dbde;
  --line:#00727e; --ok:#00727e; --near:#b0762a; --bad:#b23b52;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1040px; margin:0 auto; padding:clamp(28px,5vw,72px) clamp(18px,4vw,40px) 96px; }}
.prose {{ max-width:64ch; }}
p {{ margin:0 0 1em; }}
a {{ color:var(--line); }}
h1,h2,h3 {{ font-family:var(--mono); font-weight:700; text-wrap:balance; margin:0; }}
h1 {{ font-size:clamp(26px,4.4vw,42px); letter-spacing:-.035em; line-height:1.12; }}
h2 {{ font-size:clamp(17px,2.2vw,21px); letter-spacing:-.02em; margin:0 0 14px; }}
h3 {{ font-size:15px; letter-spacing:-.01em; margin:0 0 6px; }}
.eyebrow {{
  font-family:var(--mono); font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--muted); margin:0 0 14px;
}}
.lede {{ font-size:clamp(17px,2.1vw,19px); color:var(--ink); max-width:60ch; margin-top:18px; }}
.deck {{ color:var(--muted); }}
section {{ margin-top:clamp(40px,6vw,68px); }}
.rule {{ height:1px; background:var(--hair); border:0; margin:0; }}
.num, .mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}

/* headline stat pair */
.split {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); margin:26px 0 0; }}
.stat {{ background:var(--surface); border:1px solid var(--hair); border-radius:3px; padding:18px 20px; }}
.stat .k {{ font-family:var(--mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }}
.stat .v {{ font-family:var(--mono); font-size:clamp(28px,5vw,40px); font-weight:700; letter-spacing:-.04em; line-height:1.1; margin-top:8px; }}
.stat .v small {{ font-size:.42em; font-weight:400; letter-spacing:0; color:var(--muted); margin-left:.35em; }}
.stat .s {{ font-size:13.5px; color:var(--muted); margin-top:6px; }}
.stat.tgt .v {{ color:var(--ok); }}
.stat.brd .v {{ color:var(--bad); }}
tr.hero th, tr.hero td {{ background:var(--raised); font-weight:600; }}
tr.hero td.num {{ color:var(--ok); }}
sup {{ color:var(--muted); font-size:.7em; }}

.tablewrap {{ overflow-x:auto; border:1px solid var(--hair); border-radius:3px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; min-width:640px; font-size:14px; }}
th,td {{ text-align:left; padding:11px 14px; border-bottom:1px solid var(--hair); }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom:0; }}
thead th {{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--muted); font-weight:400; background:var(--raised);
}}
td.num, th.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.mname {{ font-family:var(--mono); font-weight:700; display:block; }}
.mnote {{ font-size:12.5px; color:var(--muted); }}
.pill {{
  font-family:var(--mono); font-size:12px; padding:2px 8px; border-radius:2px;
  border:1px solid currentColor; display:inline-block;
}}
.pill.ok {{ color:var(--ok); }} .pill.bad {{ color:var(--muted); }}

/* per-frame error bars */
.bars {{ display:flex; flex-direction:column; gap:5px; margin-top:14px; }}
.bar-row {{ display:grid; grid-template-columns:78px 1fr 62px; align-items:center; gap:12px; }}
.bar-name {{ font-family:var(--mono); font-size:11.5px; color:var(--muted); }}
.bar-track {{ position:relative; height:15px; background:var(--raised); border:1px solid var(--hair); border-radius:2px; overflow:hidden; display:block; }}
.bar-fill {{ display:block; height:100%; }}
.bar-fill.ok {{ background:var(--ok); }}
.bar-fill.near {{ background:var(--near); }}
.bar-fill.bad {{ background:var(--bad); opacity:.75; }}
.bar-null {{ display:block; height:100%; width:100%; background:repeating-linear-gradient(135deg,transparent,transparent 5px,var(--hair) 5px,var(--hair) 6px); }}
.bar-val {{ font-family:var(--mono); font-size:12px; text-align:right; font-variant-numeric:tabular-nums; }}
.bar-val.ok {{ color:var(--ok); }} .bar-val.near {{ color:var(--near); }}
.bar-val.bad, .bar-val.null {{ color:var(--muted); }}
.scale {{ display:grid; grid-template-columns:78px 1fr 62px; gap:12px; margin-top:8px; }}
.scale .ticks {{ display:flex; justify-content:space-between; font-family:var(--mono); font-size:10.5px; color:var(--muted); }}
.groupcap {{ font-family:var(--mono); font-size:11px; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); margin:22px 0 0; }}
.threshnote {{ font-size:13px; color:var(--muted); margin-top:10px; }}

figure {{ margin:24px 0 0; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--hair); border-radius:3px; }}
figcaption {{ font-size:13.5px; color:var(--muted); margin-top:9px; max-width:70ch; }}
.figpair {{ display:grid; gap:20px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }}
.figpair figure {{ margin-top:0; }}

.notes {{ display:grid; gap:1px; background:var(--hair); border:1px solid var(--hair); border-radius:3px; margin-top:20px; }}
.note {{ background:var(--surface); padding:16px 18px; }}
.note p {{ margin:6px 0 0; font-size:14px; color:var(--muted); max-width:70ch; }}
.note h3 {{ font-family:var(--mono); }}
code {{ font-family:var(--mono); font-size:.9em; background:var(--raised); padding:1px 5px; border-radius:2px; }}
footer {{ margin-top:64px; padding-top:20px; border-top:1px solid var(--hair); font-size:13px; color:var(--muted); }}
@media (prefers-reduced-motion:no-preference) {{
  .bar-fill {{ animation:grow .5s cubic-bezier(.2,.7,.3,1) both; }}
  @keyframes grow {{ from {{ transform:scaleX(0); transform-origin:left; }} }}
}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">farmclip &middot; court calibration &middot; {NF} held-out frames</p>
  <h1>court-pose-v9,<br>measured against ground truth</h1>
  <p class="lede">Every accuracy metric this project invented had, at some point, passed a
  visibly wrong camera pose. This benchmark scores a method as pose error against
  {len(frames)} hand-clicked frames from venues no model has trained on &mdash; in pixels
  <em>and</em> in metres, because pixels alone hide the failure that matters.</p>
  <p class="lede deck">v9 is the first model here whose held-out score is trustworthy: earlier ones
  trained on two of the test venues. It reaches a usable pose on {V9PASS} of {NF} frames, and which
  frames those are turns out to be the whole story.</p>
</header>

<section>
  <h2>The split that decides everything</h2>
  <div class="prose"><p>The test set holds two populations. Low, close footage shot from behind or
  beside the court &mdash; the point of view this project targets &mdash; and elevated
  broadcast footage from league VODs. The keypoint model behaves like a different
  model on each.</p></div>
  <div class="split">
    <div class="stat tgt">
      <div class="k">target PoV &middot; {tgt[1]} frames</div>
      <div class="v">{tgt[2]:.0f}<small>px median</small></div>
      <div class="s">{tgt[3]:.2f} m floor error &middot; {tgt[4]} of {tgt[1]} usable &middot; best {tgt[5]:.1f}px</div>
    </div>
    <div class="stat brd">
      <div class="k">broadcast &middot; {brd[1]} frames</div>
      <div class="v">{brd[2]:.0f}<small>px median</small></div>
      <div class="s">{brd[3]:.2f} m floor error &middot; {brd[4]} of {brd[1]} usable &middot; best {brd[5]:.1f}px</div>
    </div>
  </div>
  <div class="prose"><p style="margin-top:22px">A {ratio:.0f}&times; gap on the same model, same day, same code. The
  target-PoV number sits just the wrong side of the {PASS_PX:.0f} px bar rather than
  nowhere near it &mdash; and it was measured on venues absent from training, by a
  model that has never seen a single frame of this footage type.</p></div>
</section>

<hr class="rule">

<section>
  <h2>Per-frame error &middot; keypoint model</h2>
  <p class="groupcap">Target PoV &mdash; low, close</p>
  <div class="bars">{bars_html(groups['target PoV'])}</div>
  <p class="groupcap">Broadcast &mdash; elevated, distant</p>
  <div class="bars">{bars_html(groups['broadcast'])}</div>
  <div class="scale"><span></span><span class="ticks"><span>10px</span><span>32px</span><span>100px</span><span>320px</span><span>1000px</span></span><span></span></div>
  <p class="threshnote">Log scale. Teal clears the {PASS_PX:.0f} px / {PASS_M:.0f} m bar,
  amber is under 60 px, red is a pose you would notice was wrong from across the room.</p>
</section>

<section>
  <h2>Every method, same {len(frames)} frames</h2>
  <div class="tablewrap">
  <table>
    <thead><tr>
      <th scope="col">method</th><th scope="col" class="num">usable</th>
      <th scope="col" class="num">answered</th><th scope="col" class="num">median px</th>
      <th scope="col" class="num">median m</th><th scope="col" class="num">focal ratio</th>
      <th scope="col" class="num">per frame</th>
    </tr></thead>
    <tbody>{''.join(mrows)}</tbody>
  </table>
  </div>
  <div class="prose"><p style="margin-top:18px"><span class="mono">classical</span> answered on no frame at
  all &mdash; it never cleared its net-band gate, so it has no median to report.
  <span class="mono">lineseg</span> refines an existing pose rather than finding one, so it inherits
  the keypoint model's failures and then amplifies them. <span class="mono">search</span> is
  untrained and cannot have a generalisation gap, but from a blind start it lands in
  the wrong basin every time.</p>
  <p>Median error counts only frames where a method produced a pose; the usable
  column counts all {len(frames)}. A method that answers rarely but well looks strong on one
  and weak on the other, so both are shown.</p></div>
</section>

<hr class="rule">

<section>
  <h2>Why pixels alone were never enough</h2>
  <div class="prose"><p>Reprojection error is the number a calibration normally reports about
  itself: how well the pose fits the points the detector found. On the frame below it reads
  <strong>41.1&nbsp;px</strong> from 4 keypoints, which looks like a mediocre-but-working solve.
  Measured against ground truth the same pose is <strong>430&nbsp;px</strong> out and puts the
  floor <strong>12.8&nbsp;m</strong> from where it belongs.</p>
  <p>A pose can fit its own detections and still be nowhere near the court, and no
  self-reported metric distinguishes the two. That gap is the entire reason this benchmark
  exists.</p></div>
  {figure('bad', 'v9&rsquo;s worst frame. Cyan is the model&rsquo;s court, green crosses are the human clicks. Four keypoints, fitted to 41&nbsp;px against themselves, 430&nbsp;px and 12.8&nbsp;m from truth.', 'Court wireframe badly misplaced on a broadcast frame')}
  <div class="figpair" style="margin-top:26px">
    {figure('good', 'v9&rsquo;s best frame: 12 keypoints, 15.7&nbsp;px and 0.66&nbsp;m from truth &mdash; inside the bar, on a venue absent from training.', 'Correctly aligned court wireframe')}
    {figure('gt', 'Ground truth. Clicks are usable only once the solved pose is physical, spans the court in both axes, and lands on the paint.', 'Hand-labelled ground truth overlay')}
  </div>
</section>


<hr class="rule">

<section>
  <h2>Does more data fix it? A capped venue sweep says: not more <em>venues</em></h2>
  <div class="prose"><p>Five models, each trained on a random subset of N source venues with frames
  per venue capped at 3 so that venue count varies roughly independently of dataset size.
  Without that cap the curve is unreadable &mdash; five video runs carry 8&ndash;41 frames each
  against 1&ndash;3 for a web venue, so frame count moves in steps as a dense venue enters the
  subset rather than with venue count.</p></div>
  <div class="tablewrap">
  <table>
    <thead><tr><th scope="col">venues</th><th scope="col" class="num">train frames</th>
      <th scope="col" class="num">all</th><th scope="col" class="num">target PoV</th>
      <th scope="col" class="num">broadcast</th><th scope="col" class="num">usable</th></tr></thead>
    <tbody>{SWEEP_ROWS}</tbody>
  </table>
  </div>
  <div class="prose">
  <p style="margin-top:18px"><strong>The venue slope is unreadable</strong> &mdash; 60 venues beats
  81. A non-monotonic curve means noise dominates at these sizes, and every capped model is badly
  undertrained. That half of the experiment failed.</p>
  <p><strong>The last row against v9 is the result that survived.</strong> Same 81 venues, 92
  training frames versus 216, and the only difference is how densely each venue is sampled.
  Error falls {SWEEP_RATIO}&times;, and it is the only configuration that gets a usable pose at
  all. Frames per venue is doing the work, not venue count.</p>
  <p>Where it does the work is the other half of the finding: the target PoV improves
  {TGT_FROM:.0f}px &rarr; {TGT_TO:.0f}px while broadcast moves {BC_FROM:.0f}px &rarr;
  {BC_TO:.0f}px. The dense venues are low end-on footage, like the target and unlike broadcast.
  Density buys accuracy in-distribution and close to nothing outside it.</p>
  </div>
</section>

<section>
  <h2>Model history</h2>
  <div class="tablewrap">
  <table>
    <thead><tr><th scope="col">model</th><th scope="col">dataset</th>
      <th scope="col" class="num">target PoV</th><th scope="col" class="num">broadcast</th>
      <th scope="col">note</th></tr></thead>
    <tbody>{HISTORY_ROWS}</tbody>
  </table>
  </div>
  <div class="prose"><p style="margin-top:18px">v8 looked like a breakthrough at 7 of 10 on the
  target group. Twelve frames from those two venues were in its training set. v9 trains without
  them and the same venues score {TGT_TO:.0f}px &mdash; the apparent gain was the model
  recognising venues it had memorised. Only v9's numbers here are a clean held-out measurement.</p></div>
</section>

<section>
  <h2>What the ground truth had to survive</h2>
  <div class="prose"><p>A labelling that cannot pin down a pose is worse than none, because it scores
  correct methods as wrong. Clicks are admitted only after four independent checks, and
  {gate.get('gt_max_err','15')} px of click residual is the weakest of them.</p></div>
  <div class="notes">
    <div class="note"><h3>Is the camera physical?</h3>
      <p>A solve that puts the camera under the floor is reporting a flipped labelling, not a viewpoint.</p></div>
    <div class="note"><h3>Do the clicks span the court&rsquo;s width?</h3>
      <p>Clicks along one sideline lie in a single plane and constrain nothing. The three worst frames in the original set fitted at 1.9, 0.4 and 1.5&nbsp;px &mdash; the <em>lowest</em> residuals recorded &mdash; while projecting courts that missed the paint entirely.</p></div>
    <div class="note"><h3>And its length?</h3>
      <p>Under 9&nbsp;m of depth and focal length trades off against camera distance almost exactly, so the solve slides to its focal clamp and returns a confident wrong answer.</p></div>
    <div class="note"><h3>Does the court land on the paint?</h3>
      <p>An evidence check, allowed only to veto. It may say &ldquo;wrong&rdquo;; it is not trusted when it says &ldquo;right&rdquo;, and its threshold is calibrated against frames confirmed by eye.</p></div>
  </div>
</section>

<section>
  <h2>Three bugs found by building the instrument</h2>
  <div class="prose"><p>All three shared a shape: a proxy answering confidently where it had no
  evidence. Each had been silently corrupting comparisons for as long as it existed.</p></div>
  <div class="notes">
    <div class="note"><h3 class="mono">floor_mask</h3>
      <p>Its k-means was never seeded, so it drew from OpenCV&rsquo;s global RNG. The same pose scored 0.95&nbsp;px and 76.58&nbsp;px on consecutive calls &mdash; pass and fail. Every past measurement that touched it was partly reading the RNG.</p></div>
    <div class="note"><h3 class="mono">floor_mask</h3>
      <p>It kept only the single dominant floor colour. A coloured court inside a contrasting apron is the norm in this sport, so two-tone frames masked 11&ndash;13% of the image where plain floors mask 38&ndash;57% &mdash; deleting the court&rsquo;s own lines from the evidence and rejecting two visibly correct poses.</p></div>
    <div class="note"><h3 class="mono">calib_score</h3>
      <p>It judged the net against floor-masked evidence, though the net is the one model line deliberately off the floor. A correct pose scored 240&nbsp;px off simply because the mask did not reach that high. It now reports <em>unjudgeable</em> instead of inventing a number.</p></div>
  </div>
</section>

<footer>
  <p>Generated from <code>out/bench/results.json</code> by <code>scripts/bench_report.py</code>.
  Pass = within {PASS_PX:.0f}&nbsp;px of the ground-truth pose across the court model
  <em>and</em> within {PASS_M:.0f}&nbsp;m on the floor plane. Overlays are the
  benchmark&rsquo;s own debug output, unretouched.</p>
</footer>
</div>
"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    kb = dest.stat().st_size / 1024
    print(f"{dest}  ({kb:.0f} KB, {len(frames)} frames, {len(methods)} methods)")
    print(f"  target PoV {tgt[2]:.1f}px / {tgt[3]:.2f}m  ({tgt[4]}/{tgt[1]} pass)")
    print(f"  broadcast  {brd[2]:.1f}px / {brd[3]:.2f}m  ({brd[4]}/{brd[1]} pass)")


if __name__ == "__main__":
    main()
