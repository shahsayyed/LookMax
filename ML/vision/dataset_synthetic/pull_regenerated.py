#!/usr/bin/env python3
"""
pull_regenerated.py -- downloads newly (re)generated images from a remote box
back to the local raw_generated/images/ as they complete, instead of waiting
for the whole run to finish. Companion to regenerate_targeted.py.

Deliberately NOT fleet_monitor.py's pull_images_from_host(): that pull uses
--ignore-existing because it was built for the main production run, where a
pulled filename is always NEW (never seen locally before). Here every target
filename already exists locally (that's the whole point -- we're replacing a
bad image with a fixed one), so --ignore-existing would silently skip pulling
every single regenerated image. This pull overwrites intentionally, but ONLY
for filenames in the given whitelist -- never a directory-wide sync.

Progress is tracked by what rsync itself confirms it actually transferred
(--out-format), NOT by local file existence -- the target files already exist
locally under their old, wrong pixels before any pull happens, so an
existence check would (and, in an earlier version of this script, did)
falsely read "already done" on the very first poll. Confirmed-transferred
filenames accumulate in a small JSON state file next to the target list, so
progress survives this script being killed/restarted (the remote instances
in this project have not proven reliable) -- a completed file is never
re-counted as pending even across a fresh invocation.

Usage:
  python3 pull_regenerated.py --host vast1 --targets regen_targets.txt              # one pull
  python3 pull_regenerated.py --host vast1 --targets regen_targets.txt --watch 120  # poll every 120s until all 1460 land
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
LOCAL_IMAGES_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "raw_generated" / "images"
REMOTE_IMAGES_DIR = "/data/qwen_dataset_output/images/"


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _state_path(targets_path):
    return Path(targets_path).with_suffix(".pulled_state.json")


def load_confirmed(targets_path):
    p = _state_path(targets_path)
    if p.exists():
        return set(json.loads(p.read_text()))
    return set()


def save_confirmed(targets_path, confirmed):
    _state_path(targets_path).write_text(json.dumps(sorted(confirmed)))


RSYNC_BASE_FLAGS = [
    "-rtz", "--checksum",   # -rtz (NOT -a): content + mtime only, deliberately no owner/group
                             # preservation -- pulling from a root-owned Linux box to a macOS user
                             # account can never satisfy an owner match, so -a made rsync think an
                             # already-correct file still needed transferring, every single poll.
                             # --checksum over size+mtime because the quick-check separately
                             # under-detected real content changes in this exact pipeline.
    "--exclude=*.tmp",       # never pull an in-progress atomic-write temp file
    "--min-size=1",          # never pull a zero-byte file
]


def pull_once(host, targets_path, dest_dir, confirmed):
    """One rsync pass, restricted to exactly the filenames in targets_path.
    Mutates `confirmed` in place with every filename rsync reports as
    actually transferred this pass. With -rtz --checksum (content-only
    comparison, no owner/group -- see RSYNC_BASE_FLAGS), a file only gets
    (re-)printed by --out-format when its content genuinely still differs,
    so accumulating printed filenames across polls is a correct, cumulative
    "confirmed done" set: a file not yet generated on the remote at all
    simply produces a link_stat error (ignored) and is never printed, so it
    correctly stays un-confirmed until the remote actually has it.
    Returns len(confirmed) or None on a transport failure (host unreachable,
    still booting, etc -- not fatal, the caller just retries on the next poll)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    rsync_cmd = (
        ["rsync"] + RSYNC_BASE_FLAGS + [
            "--files-from=" + str(targets_path),
            "--out-format=%n",       # print exactly the filenames actually transferred
            "-e", "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no",
            f"{host}:{REMOTE_IMAGES_DIR}",
            str(dest_dir) + "/",
        ]
    )
    try:
        proc = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        print(f"[{_ts()}] pull failed (transport): {e}")
        return None
    # rsync exit 23: some files vanished/weren't there yet (remote hasn't
    # generated them yet) -- expected mid-run, not a real failure.
    if proc.returncode not in (0, 23, 24):
        print(f"[{_ts()}] pull failed rc={proc.returncode}: {proc.stderr.strip()[:200]}")
        return None

    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            confirmed.add(line)
    return len(confirmed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="SSH alias/host (e.g. vast1)")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--dest", default=str(LOCAL_IMAGES_DIR))
    parser.add_argument("--watch", type=int, default=0, help="Poll every N seconds until all targets are confirmed pulled (0 = pull once and exit)")
    parser.add_argument("--reset", action="store_true", help="Ignore any saved progress and start counting from zero")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    total = len([l for l in Path(args.targets).read_text().splitlines() if l.strip()])
    confirmed = set() if args.reset else load_confirmed(args.targets)
    if confirmed:
        print(f"Resuming with {len(confirmed)}/{total} already confirmed pulled from a previous run.")

    if args.watch <= 0:
        have = pull_once(args.host, args.targets, dest_dir, confirmed)
        save_confirmed(args.targets, confirmed)
        if have is None:
            sys.exit(1)
        print(f"[{_ts()}] {have}/{total} regenerated images confirmed pulled.")
        return

    print(f"Polling {args.host} every {args.watch}s until all {total} regenerated images are confirmed pulled...")
    last_have = len(confirmed) - 1  # force the first print
    while True:
        have = pull_once(args.host, args.targets, dest_dir, confirmed)
        if have is not None:
            if have != last_have:
                print(f"[{_ts()}] {have}/{total} confirmed pulled so far.")
                save_confirmed(args.targets, confirmed)
                last_have = have
            if have >= total:
                print(f"[{_ts()}] All {total} regenerated images confirmed pulled.")
                return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
