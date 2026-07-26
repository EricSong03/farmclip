Set-Location C:\Users\ericy\projects\farmclip
$log = "out\fullrun.log"
"[$(Get-Date -Format HH:mm:ss)] start" | Out-File $log -Encoding utf8
$menlo = Get-ChildItem "examples\Long Beach*.mp4" | Select-Object -First 1
$mikasa = Get-ChildItem "examples\First Game*.mp4" | Select-Object -First 1
python -m uv run farmclip $menlo.FullName --start 0 --end 60 --out out --player-model yolo11s.pt --player-imgsz 1280 --per-side 6 2>&1 | Out-File $log -Append -Encoding utf8
python -m uv run farmclip $mikasa.FullName --start 60 --end 720 --out out/mikasa --player-model yolo11n.pt --player-imgsz 960 --per-side 2 --player-step 3 2>&1 | Out-File $log -Append -Encoding utf8
python -m uv run python scripts/metrics.py out 12 2>&1 | Out-File $log -Append -Encoding utf8
python -m uv run python scripts/metrics.py out/mikasa 4 2>&1 | Out-File $log -Append -Encoding utf8
"[$(Get-Date -Format HH:mm:ss)] DONE" | Out-File $log -Append -Encoding utf8
