"""Fail if anything under data/ is ignored or untracked.

This exists because the same failure has now happened three times, each time
silently:

  1. Labelled frames were staged into out/webimgs, which is gitignored. They
     were never committed, and court-pose-v7 trained without them.
  2. The annotations for four dome venues were never tracked at all -- hand
     clicks that existed on exactly one machine.
  3. The migration that was supposed to fix (1) and (2) moved them to
     data/runs/ -- which .gitignore's unanchored `runs/` pattern matched, so
     `git add -A data` skipped the directory without a word.

Every one of those was invisible in `git status`, because git does not report
files it has been told to ignore. The only reliable way to notice is to ask
git directly about the paths that matter, which is what this does.

Usage: python -m uv run python scripts/check_data.py
Exit code 1 on any problem, so it can gate a commit or a training run.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# Directories whose contents are legitimately absent from git: rebuildable from
# the tracked pool with a seeded command, and large.
ALLOWED_IGNORED = ("data/dataset/court_v", "data/dataset/court_c")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.splitlines()


def main():
    if not DATA.exists():
        sys.exit("data/ does not exist")

    on_disk = {p.relative_to(ROOT).as_posix()
               for p in DATA.rglob("*") if p.is_file()}
    tracked = set(git("ls-files", "data"))
    missing = sorted(on_disk - tracked)
    allowed = [m for m in missing if m.startswith(ALLOWED_IGNORED)]
    problems = [m for m in missing if not m.startswith(ALLOWED_IGNORED)]

    # Distinguish "ignored" from "merely untracked" -- ignored is the dangerous
    # one, because `git status` stays silent about it.
    ignored = set(git("check-ignore", *problems)) if problems else set()

    print(f"data/: {len(on_disk)} files on disk, {len(tracked)} tracked")
    if allowed:
        print(f"  {len(allowed)} rebuildable and deliberately untracked "
              f"({ALLOWED_IGNORED[0]}*)")

    if not problems:
        print("check_data ok: everything under data/ is tracked")
        return 0

    for m in problems[:40]:
        why = "IGNORED by .gitignore" if m in ignored else "untracked"
        print(f"  {why}: {m}")
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more")
    print(f"\n{len(problems)} file(s) under data/ are not in git"
          f"{f', {len(ignored)} of them silently ignored' if ignored else ''}.")
    print("This is data that exists on one machine only. Commit it, or add an "
          "explicit exemption to ALLOWED_IGNORED with a reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
