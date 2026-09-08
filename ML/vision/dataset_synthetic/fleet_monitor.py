#!/usr/bin/env python3
"""
fleet_monitor.py -- Automated health, progress monitor, auto-destroy controller,
and cross-machine zero-byte coordination sync for LookMax synthetic generation fleet.

Features:
- Queries image counts, tmux status, and GPU utilization via SSH.
- Tracks progress and idle time across checks in fleet_state.json.
- Logs history to fleet_monitor.log and updates fleet_status.md.
- Alerts when a machine has stopped or made zero progress for >= 3 hours.
- AUTOMATIC DESTRUCTION UPON COMPLETION:
  When a machine completes 100% of its target images:
    1. Runs a final verified rsync to local Mac storage (/ML/data/vision_synthetic/raw_generated/).
    2. Verifies local data integrity and ensures 0 .tmp files.
    3. Calls Vast.ai REST API to destroy the instance so billing stops immediately.
    4. Alerts the user with confirmation of safe completion and destruction.
- ZERO-BYTE CROSS-MACHINE COORDINATION (--sync-loop):
  Every 5 minutes:
    1. Pulls real completed .png (min-size=1, no .tmp) from each machine -> local Mac.
    2. For every image now on local Mac, creates a zero-byte marker in a staging dir.
    3. Pushes those zero-byte markers to every OTHER reachable machine with
       --ignore-existing, so a machine that already has the real file is unaffected,
       but a machine that has not generated it yet will skip it (already_done_indices
       in full_run.py skips any .png that exists, regardless of size).
  This prevents duplicate generation if two machines are ever assigned overlapping slices.

Usage:
  python3 fleet_monitor.py                     # Single manual status check
  python3 fleet_monitor.py --cron              # Silent cron check; exit 2 if alert, exit 0 if normal
  python3 fleet_monitor.py --loop              # Background loop running every hour (3600s)
  python3 fleet_monitor.py --sync-loop         # Continuous 5-min pull+push sync (run as daemon)
  python3 fleet_monitor.py --sync-once         # Run one sync cycle then exit
  python3 fleet_monitor.py --set-api-key <KEY> # Save Vast.ai API key securely
  python3 fleet_monitor.py --test-api-key      # Validate Vast.ai API key and list active instances
"""

import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
STATE_FILE = SCRIPT_DIR / "fleet_state.json"
LOG_FILE = SCRIPT_DIR / "fleet_monitor.log"
SYNC_LOG_FILE = SCRIPT_DIR / "fleet_sync.log"
STATUS_MD = SCRIPT_DIR / "fleet_status.md"
API_KEY_FILE = SCRIPT_DIR / ".vast_api_key"
LOCAL_DEST_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "raw_generated"
# Staging dir for zero-byte markers — never touches real images
MARKERS_DIR = LOCAL_DEST_DIR / ".markers"

HOSTS = {
    "vast1": {
        "slice": "0:28000",
        "target": 2236,
        "instance_id": 50316397,
        "desc": "1x H100 SXM (80GB)"
    }
}

STALL_ALERT_HOURS = 0.5      # Alert and auto-destroy if interrupted/unreachable for >= 30 minutes
SYNC_INTERVAL_SECS = 30      # 30 seconds between sync cycles

REMOTE_CHECK_CMD = r"""
CONFIG="/data/LookMax_Generator/.auto_resume_config"
TASK_RANGE=""
[ -f "$CONFIG" ] && source "$CONFIG"
IMG_COUNT=$(find /data/qwen_dataset_output/images -maxdepth 1 -name "*.png" ! -name "*.tmp" ! -size 0 2>/dev/null | wc -l | tr -d " ")
TMUX_STATE=$(tmux has-session -t lookmax_gen 2>/dev/null && echo "RUNNING" || echo "STOPPED")
GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "")
echo "COUNT=$IMG_COUNT"
echo "TMUX=$TMUX_STATE"
echo "TASK_RANGE=$TASK_RANGE"
echo "GPU_START"
echo "$GPU_STATS"
echo "GPU_END"
"""


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

def get_vast_api_key() -> str | None:
    env_key = os.environ.get("VAST_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    if API_KEY_FILE.exists():
        try:
            k = API_KEY_FILE.read_text().strip()
            if k:
                return k
        except Exception:
            pass
    home_file = Path.home() / ".vast_api_key"
    if home_file.exists():
        try:
            k = home_file.read_text().strip()
            if k:
                return k
        except Exception:
            pass
    return None


def set_vast_api_key(key: str):
    API_KEY_FILE.write_text(key.strip() + "\n")
    os.chmod(API_KEY_FILE, 0o600)
    print(f"✓ Vast.ai API key saved securely to {API_KEY_FILE}")


def test_vast_api_key(key: str | None = None) -> bool:
    api_key = key or get_vast_api_key()
    if not api_key:
        print("✗ No Vast.ai API key found. Use --set-api-key <KEY> to save one.")
        return False
    url = f"https://console.vast.ai/api/v1/instances/?api_key={api_key}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            instances = data.get("instances", [])
            print(f"✓ Vast.ai API key valid! Found {len(instances)} active instance(s):")
            for inst in instances:
                iid = inst.get("id")
                ip = inst.get("public_ipaddr") or inst.get("ssh_host")
                ports = inst.get("ports", {})
                ssh_port = ports.get("22/tcp", [{}])[0].get("HostPort", inst.get("ssh_port"))
                status = inst.get("actual_status")
                gpu_name = inst.get("gpu_name")
                num_gpus = inst.get("num_gpus")
                cost = inst.get("dph_total", 0.0)
                print(f"  - Instance ID: {iid} | Status: {status} | Host: {ip}:{ssh_port} | {num_gpus}x {gpu_name} (${cost:.3f}/hr)")
            return True
    except urllib.error.HTTPError as e:
        print(f"✗ Vast.ai API error: HTTP {e.code} ({e.reason})")
        return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False


# ---------------------------------------------------------------------------
# Vast.ai instance management
# ---------------------------------------------------------------------------

def destroy_vast_instance(instance_id: int, api_key: str) -> bool:
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/?api_key={api_key}"
    req = urllib.request.Request(url, method="DELETE", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("success", False) or resp.status in (200, 204)
    except Exception as e:
        print(f"Error calling Vast.ai DELETE on instance {instance_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Final completion sync (full verified pull — unchanged)
# ---------------------------------------------------------------------------

def sync_and_verify_host(host: str, expected_count: int) -> bool:
    LOCAL_DEST_DIR.mkdir(parents=True, exist_ok=True)
    images_dest = LOCAL_DEST_DIR / "images"
    images_dest.mkdir(parents=True, exist_ok=True)

    print(f"==> [{host}] Target reached! Initiating final data sync to {LOCAL_DEST_DIR}...")
    rsync_cmd = [
        "rsync", "-avz",
        "--exclude=*.tmp",
        "-e", "ssh -S none -o ConnectTimeout=15 -o StrictHostKeyChecking=no",
        f"{host}:/data/qwen_dataset_output/",
        f"{LOCAL_DEST_DIR}/"
    ]
    try:
        proc = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f"✗ rsync failed with code {proc.returncode}: {proc.stderr[:100]}")
            return False
    except Exception as e:
        print(f"✗ rsync error: {e}")
        return False

    local_images = list(images_dest.glob("*.png"))
    local_tmps = list(images_dest.glob("*.tmp"))
    if local_tmps:
        print(f"✗ Warning: {len(local_tmps)} .tmp files found in local storage!")
        return False
    print(f"✓ Local archive verified: {len(local_images)} total valid images present in {images_dest}.")
    return True


# ---------------------------------------------------------------------------
# Zero-byte cross-machine coordination sync
# ---------------------------------------------------------------------------

def _sync_log(msg: str):
    """Append a timestamped line to fleet_sync.log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SYNC_LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def is_host_reachable(host: str) -> bool:
    """Quick SSH probe — returns True only if shell responds."""
    try:
        proc = subprocess.run(
            ["ssh", "-S", "none", "-o", "ConnectTimeout=8",
             "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             host, "echo ok"],
            capture_output=True, text=True, timeout=12
        )
        return proc.returncode == 0 and "ok" in proc.stdout
    except Exception:
        return False


def pull_images_from_host(host: str, images_dest: Path) -> int:
    """
    Pull real completed images (min-size=1, exclude .tmp) from remote -> local.

    --ignore-existing: a real local image is NEVER overwritten (protects against
    a race where the remote somehow has a zero-byte file for a name we already
    have locally as a full image).

    Returns number of newly added files, or -1 on failure.
    """
    images_dest.mkdir(parents=True, exist_ok=True)
    before = {f.name for f in images_dest.iterdir()
              if f.is_file() and not f.name.endswith(".tmp")}

    rsync_cmd = [
        "rsync", "-az",
        "--exclude=*.tmp",    # never pull in-progress temp files
        "--min-size=1",       # never pull zero-byte markers we previously pushed
        "--ignore-existing",  # never overwrite a real local image
        "-e", "ssh -S none -o ConnectTimeout=15 -o StrictHostKeyChecking=no",
        f"{host}:/data/qwen_dataset_output/images/",
        str(images_dest) + "/"
    ]
    try:
        proc = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            _sync_log(f"PULL {host} FAILED rc={proc.returncode}: {proc.stderr.strip()[:120]}")
            return -1
    except Exception as e:
        _sync_log(f"PULL {host} exception: {e}")
        return -1

    after = {f.name for f in images_dest.iterdir()
             if f.is_file() and not f.name.endswith(".tmp")}
    return len(after - before)


FILENAME_TO_POS_MAP = None

def get_filename_to_pos_map() -> dict:
    global FILENAME_TO_POS_MAP
    if FILENAME_TO_POS_MAP is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            from full_run import build_full_task_list
            tasks = build_full_task_list()
            FILENAME_TO_POS_MAP = {t["filename"]: i for i, t in enumerate(tasks)}
        except Exception:
            FILENAME_TO_POS_MAP = {}
    return FILENAME_TO_POS_MAP


def is_filename_in_host_slice(filename: str, host: str) -> bool:
    """Check if filename's task list position (0..27999) falls inside host's assigned slice."""
    meta = HOSTS.get(host)
    if not meta or "slice" not in meta:
        return False
    try:
        parts = meta["slice"].split(":")
        start_pos, end_pos = int(parts[0]), int(parts[1])
        pos_map = get_filename_to_pos_map()
        pos = pos_map.get(filename)
        if pos is not None:
            return start_pos <= pos < end_pos
    except Exception:
        pass
    return False


def push_markers_to_host(host: str, images_dest: Path, state: dict) -> tuple[int, int]:
    """
    Optimized Delta Marker Push:
    Only sends markers for images that have NOT yet been recorded as pushed to `host`,
    and NEVER pushes markers for images falling inside `host`'s own assigned task slice.

    If 0 new markers are needed for `host`, returns immediately (0.00s execution).
    If new markers exist (e.g. 5 new images), creates a temporary delta folder,
    rsyncs only those 5 files over SSH with --ignore-existing, and records them in state.
    """
    host_state = state["hosts"].setdefault(host, {})
    pushed_set = set(host_state.get("pushed_markers", []))

    # Real non-zero images saved locally on Mac
    real_images = {
        f.name for f in images_dest.iterdir()
        if f.is_file() and f.stat().st_size > 0 and not f.name.endswith(".tmp")
    }

    needed = real_images - pushed_set
    if not needed:
        return 0, 0

    delta_dir = SCRIPT_DIR / f".markers_delta_{host}"
    if delta_dir.exists():
        for p in delta_dir.iterdir():
            p.unlink(missing_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name in needed:
            (delta_dir / name).touch()

        rsync_cmd = [
            "rsync", "-az",
            "--ignore-existing",
            "-e", "ssh -S none -o ConnectTimeout=15 -o StrictHostKeyChecking=no",
            str(delta_dir) + "/",
            f"{host}:/data/qwen_dataset_output/images/"
        ]
        proc = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            pushed_set.update(needed)
            host_state["pushed_markers"] = list(pushed_set)
            save_state(state)
            return len(needed), 0
        else:
            _sync_log(f"PUSH delta to {host} failed rc={proc.returncode}: {proc.stderr.strip()[:120]}")
            return 0, proc.returncode
    finally:
        if delta_dir.exists():
            for p in delta_dir.iterdir():
                p.unlink(missing_ok=True)
            try:
                delta_dir.rmdir()
            except Exception:
                pass


def sync_markers(verbose: bool = True) -> dict:
    """
    One full sync cycle:
      Phase 1 — Pull real images from every reachable host -> local Mac.
      Phase 2 — Push ONLY delta zero-byte markers to reachable hosts (0.00s if no new files).

    Returns per-host summary dict.
    """
    images_dest = LOCAL_DEST_DIR / "images"
    images_dest.mkdir(parents=True, exist_ok=True)

    state = load_state()
    summary = {}
    reachable_hosts = []

    # Phase 1: Pull real images from all reachable hosts
    for host in HOSTS:
        if verbose:
            print(f"  [{host}] Pulling...", end=" ", flush=True)
        reachable = is_host_reachable(host)
        summary[host] = {"reachable": reachable, "pulled": 0, "markers_sent": 0, "push_rc": None}
        if not reachable:
            _sync_log(f"PULL {host}: unreachable — skipped")
            if verbose:
                print("unreachable")
            continue
        reachable_hosts.append(host)
        n = pull_images_from_host(host, images_dest)
        summary[host]["pulled"] = n
        msg = f"+{n} new" if n >= 0 else "FAILED"
        _sync_log(f"PULL {host}: {msg}")
        if verbose:
            print(msg)

    total_local = sum(
        1 for f in images_dest.iterdir()
        if f.is_file() and f.stat().st_size > 0 and not f.name.endswith(".tmp")
    )

    # Phase 2: Push ONLY delta markers to every reachable host
    for host in reachable_hosts:
        sent, rc = push_markers_to_host(host, images_dest, state)
        summary[host]["markers_sent"] = sent
        summary[host]["push_rc"] = rc
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        if sent > 0 or rc != 0:
            _sync_log(f"PUSH markers->{host}: {status} ({sent} new delta markers sent to remote)")
            if verbose:
                print(f"  [{host}] Pushed +{sent} new marker(s)")

    ls = get_local_dataset_summary()
    if verbose:
        print(f"  Local Mac Storage : {ls['total_real']:,} / 28,000 ({ls['pct_total']:.1f}%) real images saved locally.")
        slice_parts = [f"{s['key']}: {s['done']}/{s['target']} ({s['pct']:.1f}%)" for s in ls["slices"]]
        print(f"  Slices            : {' | '.join(slice_parts)}")

    _sync_log(f"Cycle done. Local total={total_local}. Reachable={reachable_hosts or ['none']}")
    return summary




def run_sync_loop(interval: int = SYNC_INTERVAL_SECS):
    """
    Continuous sync daemon — runs sync_markers() every `interval` seconds.
    Intended to be launched alongside the hourly --cron job.
    """
    print(f"LookMax fleet sync daemon started (interval: {interval}s = {interval//60} min)")
    print(f"Sync log: {SYNC_LOG_FILE}")
    print("Ctrl-C to stop.\n")
    _sync_log(f"Sync daemon started (interval={interval}s)")

    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Sync cycle...", flush=True)
        try:
            sync_markers(verbose=True)
        except Exception as e:
            _sync_log(f"Sync cycle ERROR: {e}")
            print(f"  ERROR: {e}")
        print(f"  Next in {interval}s...\n", flush=True)
        time.sleep(interval)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    state = {"hosts": {}, "history": []}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception:
            pass

    # Auto-detect instance ID changes for any host & reset pushed_markers for new instances
    for host_name, host_info in HOSTS.items():
        curr_id = host_info.get("instance_id")
        hdata = state.setdefault("hosts", {}).setdefault(host_name, {})
        saved_id = hdata.get("instance_id")
        if curr_id and saved_id != curr_id:
            hdata["pushed_markers"] = []
            hdata["instance_id"] = curr_id
            hdata["last_count"] = 0
            hdata["stalled_since_iso"] = None
            hdata["stalled_hours"] = 0.0
            hdata["completed"] = False
            hdata["destroyed"] = False
    return state


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Remote host querying
# ---------------------------------------------------------------------------
# Local Mac dataset progress tracking (offline-safe)
# ---------------------------------------------------------------------------

def get_local_dataset_summary() -> dict:
    """
    Scans local Mac storage (/ML/data/vision_synthetic/raw_generated/images)
    and computes slice-by-slice & category progress independent of server status.
    """
    images_dir = LOCAL_DEST_DIR / "images"
    done_indices = set()
    if images_dir.exists():
        import re
        for f in images_dir.iterdir():
            if f.is_file() and f.stat().st_size > 0 and not f.name.endswith(".tmp"):
                m = re.match(r"^(\d+)_", f.name)
                if m:
                    done_indices.add(int(m.group(1)))

    total_real = len(done_indices)

    all_tasks = []
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from full_run import build_full_task_list
        all_tasks = build_full_task_list()
    except Exception:
        pass

    slices_def = [
        ("original", "[0:4810] (Original)", 0, 4810),
        ("vast1", "[4810:10000] (vast1 slice)", 4810, 10000),
        ("vast4", "[10000:13000] (vast4 slice)", 10000, 13000),
        ("vast3", "[13000:16000] (vast3 slice)", 13000, 16000),
        ("vast2", "[16000:28000] (vast2 slice)", 16000, 28000),
    ]

    slice_stats = []
    for key, label, start_idx, end_idx in slices_def:
        if all_tasks:
            sub_tasks = all_tasks[start_idx:end_idx]
            target = len(sub_tasks)
            done_cnt = sum(1 for t in sub_tasks if t["index"] in done_indices)
        else:
            target = end_idx - start_idx
            done_cnt = sum(1 for idx in done_indices if start_idx <= idx < end_idx)

        pct = (done_cnt / target * 100.0) if target else 0.0
        rem = target - done_cnt
        slice_stats.append({
            "key": key,
            "label": label,
            "start": start_idx,
            "end": end_idx,
            "target": target,
            "done": done_cnt,
            "pct": pct,
            "remaining": rem
        })

    cat_stats = {}
    if all_tasks:
        from collections import Counter
        cat_done = Counter()
        cat_total = Counter()
        for t in all_tasks:
            cat_total[t["category"]] += 1
            if t["index"] in done_indices:
                cat_done[t["category"]] += 1
        for cat in ["Men_Grooming", "Women_Grooming", "Men_Outfit", "Women_Outfit"]:
            dn = cat_done[cat]
            tot = cat_total[cat]
            pct = (dn / tot * 100.0) if tot else 0.0
            cat_stats[cat] = {"done": dn, "total": tot, "pct": pct, "remaining": tot - dn}

    return {
        "total_real": total_real,
        "total_target": 28000,
        "pct_total": (total_real / 28000.0 * 100.0),
        "slices": slice_stats,
        "categories": cat_stats
    }


def print_local_summary():
    s = get_local_dataset_summary()
    print("================================================================================")
    print(f" LookMax Synthetic Dataset -- Local Mac Storage Summary (Offline-Safe)")
    print("================================================================================")
    print(f" Local Storage Dir : {LOCAL_DEST_DIR / 'images'}")
    print(f" Total Real Images : {s['total_real']:,} / {s['total_target']:,} ({s['pct_total']:.1f}% complete)")
    print(f" Remaining Needed  : {(s['total_target'] - s['total_real']):,} images\n")

    print(" Slice Breakdown (Local Mac Data):")
    print(f"   {'Slice Range':<28} | {'Done':<8} | {'Target':<8} | {'Progress':<8} | {'Remaining':<10}")
    print("   " + "-" * 70)
    for sl in s["slices"]:
        print(f"   {sl['label']:<28} | {sl['done']:<8} | {sl['target']:<8} | {sl['pct']:>6.1f}%  | {sl['remaining']:<10}")

    if s["categories"]:
        print("\n Category Breakdown (Local Mac Data):")
        for cat, cst in s["categories"].items():
            print(f"   - {cat:<18}: {cst['done']:>5} / {cst['total']:>5} ({cst['pct']:>5.1f}%) | Remaining: {cst['remaining']}")
    print("================================================================================\n")


# ---------------------------------------------------------------------------
# Remote host querying
# ---------------------------------------------------------------------------

def query_host(host: str) -> dict:
    cmd = [
        "ssh", "-S", "none",
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        host,
        REMOTE_CHECK_CMD
    ]
    result = {
        "reachable": False,
        "count": 0,
        "tmux": "UNKNOWN",
        "task_range": "",
        "gpus": [],
        "raw_error": ""
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        if proc.returncode != 0 and not proc.stdout.strip():
            result["raw_error"] = proc.stderr.strip()
            return result

        result["reachable"] = True
        lines = proc.stdout.strip().splitlines()
        in_gpu = False
        gpu_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("COUNT="):
                val = line.split("=", 1)[1]
                result["count"] = int(val) if val.isdigit() else 0
            elif line.startswith("TMUX="):
                result["tmux"] = line.split("=", 1)[1]
            elif line.startswith("TASK_RANGE="):
                result["task_range"] = line.split("=", 1)[1]
            elif line == "GPU_START":
                in_gpu = True
            elif line == "GPU_END":
                in_gpu = False
            elif in_gpu and line:
                gpu_lines.append(line)

        for gl in gpu_lines:
            parts = [p.strip() for p in gl.split(",")]
            if len(parts) >= 4:
                result["gpus"].append({
                    "util": f"{parts[0]}%",
                    "mem_used": f"{parts[1]}MB",
                    "mem_total": f"{parts[2]}MB",
                    "temp": f"{parts[3]}C"
                })
    except subprocess.TimeoutExpired:
        result["raw_error"] = "SSH connection timed out (15s)"
    except Exception as e:
        result["raw_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Fleet evaluation (health check + stall alert + auto-destroy)
# ---------------------------------------------------------------------------

def evaluate_fleet(cron_mode: bool = False) -> tuple[int, str]:
    state = load_state()
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alerts = []
    host_summaries = []
    log_line_parts = [f"[{now_str}]"]
    api_key = get_vast_api_key()

    for host, meta in HOSTS.items():
        host_state = state["hosts"].setdefault(host, {
            "last_count": 0,
            "last_progress_iso": now_iso,
            "stalled_since_iso": None,
            "stalled_hours": 0.0,
            "last_status": "UNKNOWN",
            "completed": False,
            "destroyed": False
        })

        if host_state.get("destroyed"):
            host_summaries.append({
                "host": host,
                "status": "DESTROYED",
                "count": meta["target"],
                "target": meta["target"],
                "percent": 100.0,
                "delta": 0,
                "stalled_hours": 0.0,
                "gpus": "N/A"
            })
            log_line_parts.append(f"{host}: DESTROYED (Done)")
            continue

        data = query_host(host)

        if not data["reachable"]:
            if host_state["stalled_since_iso"] is None:
                host_state["stalled_since_iso"] = now_iso
            stalled_dt = datetime.fromisoformat(host_state["stalled_since_iso"])
            stalled_hours = (now_dt - stalled_dt).total_seconds() / 3600.0
            host_state["stalled_hours"] = round(stalled_hours, 2)
            host_state["last_status"] = "UNREACHABLE"

            log_line_parts.append(
                f"{host}: UNREACHABLE (SSH failed: {data['raw_error'][:40]}) [Stalled: {stalled_hours:.1f}h]"
            )
            host_summaries.append({
                "host": host,
                "status": "UNREACHABLE",
                "count": host_state["last_count"],
                "target": meta["target"],
                "percent": 0.0,
                "delta": 0,
                "stalled_hours": stalled_hours,
                "gpus": "N/A"
            })

            if stalled_hours >= STALL_ALERT_HOURS:
                instance_id = meta.get("instance_id")
                if api_key and instance_id and not host_state.get("destroyed"):
                    print(f"==> Machine {host} unreachable for {stalled_hours:.1f}h. Destroying Vast.ai instance {instance_id}...")
                    destroyed = destroy_vast_instance(instance_id, api_key)
                    if destroyed:
                        host_state["destroyed"] = True
                        host_state["last_status"] = "DESTROYED"
                        alerts.append(
                            f"🚨 {host} was UNREACHABLE for {stalled_hours:.1f}h (>= 30 min threshold). "
                            f"Instance {instance_id} destroyed automatically."
                        )
                    else:
                        alerts.append(
                            f"🚨 {host} has been UNREACHABLE for {stalled_hours:.1f} hours! "
                            f"Error: {data['raw_error']}"
                        )
                else:
                    alerts.append(
                        f"🚨 {host} has been UNREACHABLE for {stalled_hours:.1f} hours! "
                        f"Error: {data['raw_error']}"
                    )
            continue

        curr_count = data["count"]
        last_count = host_state["last_count"]
        delta = curr_count - last_count
        target = meta["target"]
        pct = (curr_count / target * 100.0) if target else 0.0
        is_generating = (delta > 0)
        is_tmux_running = (data["tmux"] == "RUNNING")

        if curr_count >= target and target > 0 and not host_state.get("completed"):
            host_state["completed"] = True
            host_state["last_count"] = curr_count
            print(f"🎉 [{host}] Target reached: {curr_count}/{target} images generated!")

            sync_ok = sync_and_verify_host(host, target)
            if sync_ok:
                instance_id = meta.get("instance_id")
                if api_key and instance_id:
                    print(f"==> Destroying Vast.ai instance {instance_id} for {host}...")
                    destroyed = destroy_vast_instance(instance_id, api_key)
                    if destroyed:
                        host_state["destroyed"] = True
                        host_state["last_status"] = "DESTROYED"
                        alerts.append(
                            f"🎉 {host} completed all {curr_count}/{target} images! "
                            f"Data verified locally. Instance {instance_id} destroyed (billing stopped)."
                        )
                    else:
                        alerts.append(
                            f"⚠️ {host} completed {curr_count}/{target} images and data is verified locally, "
                            f"but Vast.ai API destroy of instance {instance_id} failed. Please destroy it manually."
                        )
                else:
                    alerts.append(
                        f"🎉 {host} completed all {curr_count}/{target} images and data is safely verified locally! "
                        f"Please destroy instance {instance_id} in your Vast.ai console."
                    )
            else:
                alerts.append(
                    f"⚠️ {host} finished generation ({curr_count}/{target}), but local verification failed! "
                    f"Instance {meta.get('instance_id')} was NOT destroyed to prevent data loss."
                )

        if is_generating:
            host_state["last_count"] = curr_count
            host_state["last_progress_iso"] = now_iso
            host_state["stalled_since_iso"] = None
            host_state["stalled_hours"] = 0.0
            host_state["last_status"] = "RUNNING"
            stalled_hours = 0.0
        else:
            if host_state["stalled_since_iso"] is None:
                host_state["stalled_since_iso"] = now_iso
            stalled_dt = datetime.fromisoformat(host_state["stalled_since_iso"])
            stalled_hours = (now_dt - stalled_dt).total_seconds() / 3600.0
            host_state["stalled_hours"] = round(stalled_hours, 2)
            host_state["last_status"] = data["tmux"]
            if not is_tmux_running and not host_state.get("completed"):
                host_state["last_status"] = "STOPPED"
            if stalled_hours >= STALL_ALERT_HOURS and not host_state.get("completed"):
                instance_id = meta.get("instance_id")
                if api_key and instance_id and not host_state.get("destroyed"):
                    print(f"==> Machine {host} stalled/stopped for {stalled_hours:.1f}h. Destroying Vast.ai instance {instance_id}...")
                    destroyed = destroy_vast_instance(instance_id, api_key)
                    if destroyed:
                        host_state["destroyed"] = True
                        host_state["last_status"] = "DESTROYED"
                        alerts.append(
                            f"🚨 {host} was STALLED for {stalled_hours:.1f}h (>= 30 min threshold). "
                            f"Instance {instance_id} destroyed automatically."
                        )
                    else:
                        alerts.append(
                            f"🚨 {host} has been STALLED for {stalled_hours:.1f} hours! "
                            f"(Count stuck at {curr_count}/{target}, tmux={data['tmux']})"
                        )
                else:
                    alerts.append(
                        f"🚨 {host} has been STALLED for {stalled_hours:.1f} hours! "
                        f"(Count stuck at {curr_count}/{target}, tmux={data['tmux']})"
                    )

        gpu_desc = "/".join([g["util"] for g in data["gpus"]]) if data["gpus"] else "N/A"
        log_line_parts.append(
            f"{host}: {curr_count}/{target} ({pct:.1f}%) [{data['tmux']}] "
            f"GPUs: {gpu_desc} (Δ+{delta}, stalled={stalled_hours:.1f}h)"
        )
        host_summaries.append({
            "host": host,
            "status": host_state["last_status"],
            "count": curr_count,
            "target": target,
            "percent": pct,
            "delta": delta,
            "stalled_hours": stalled_hours,
            "gpus": gpu_desc
        })

    save_state(state)

    log_entry = " | ".join(log_line_parts) + "\n"
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

    update_markdown_status(now_str, host_summaries, alerts)

    if alerts:
        alert_msg = "\n".join(alerts)
        if cron_mode:
            print(f"ALERT TRIGGERED:\n{alert_msg}")
        return 2, alert_msg
    else:
        summary_msg = "Fleet healthy:\n" + "\n".join([
            f" - {h['host']}: {h['count']}/{h['target']} ({h['percent']:.1f}%) "
            f"| {h['status']} | GPUs: {h['gpus']} (Δ+{h['delta']})"
            for h in host_summaries
        ])
        if not cron_mode:
            print(summary_msg)
        return 0, summary_msg


# ---------------------------------------------------------------------------
# Markdown status report
# ---------------------------------------------------------------------------

def update_markdown_status(timestamp: str, hosts: list[dict], alerts: list[str]):
    api_key = get_vast_api_key()
    api_status = "Configured (Auto-Destroy Active)" if api_key else "Missing (Manual Destroy Required)"

    lines = [
        "# LookMax Synthetic Dataset Generation -- Fleet Status\n\n",
        f"*Last Checked: {timestamp}* | *Vast.ai API Key: {api_status}*\n\n",
    ]

    if alerts:
        lines.append("> [!CAUTION]\n")
        for a in alerts:
            lines.append(f"> {a}\n")
        lines.append("\n")
    else:
        lines.append("> [!NOTE]\n> Fleet is healthy and actively generating synthetic data.\n\n")

    lines.append("| Host | Instance ID | Slice Target | Completed | Progress | GPUs | Stalled Time | Status |\n")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

    total_done = 0
    total_target = 0
    for h in hosts:
        total_done += h["count"]
        total_target += h["target"]
        stalled_str = f"{h['stalled_hours']:.1f}h" if h["stalled_hours"] > 0 else "0h (active)"
        status_badge = ("🟢 RUNNING" if h["status"] == "RUNNING"
                        else ("🏁 DESTROYED" if h["status"] == "DESTROYED"
                              else f"🔴 {h['status']}"))
        lines.append(
            f"| **{h['host']}** | `{HOSTS[h['host']]['instance_id']}` "
            f"| {HOSTS[h['host']]['slice']} ({h['target']}) | "
            f"**{h['count']}** | {h['percent']:.1f}% | {h['gpus']} "
            f"| {stalled_str} | {status_badge} |\n"
        )

    lines.append(
        f"| **TOTAL** | - | **{total_target}** | **{total_done}** "
        f"| **{(total_done/total_target*100):.1f}%** | - | - | - |\n\n"
    )

    ls = get_local_dataset_summary()
    lines.append("### Local Storage Progress (Offline-Safe)\n\n")
    lines.append(f"**Total Real Images Saved Locally on Mac**: `{ls['total_real']:,}` / `28,000` (**{ls['pct_total']:.1f}%** complete)\n\n")
    lines.append("| Slice Range | Target | Saved Locally | Progress | Remaining |\n")
    lines.append("| :--- | :--- | :--- | :--- | :--- |\n")
    for sl in ls["slices"]:
        lines.append(f"| **{sl['label']}** | {sl['target']:,} | **{sl['done']:,}** | {sl['pct']:.1f}% | {sl['remaining']:,} |\n")
    lines.append("\n")

    if ls["categories"]:
        lines.append("#### Local Category Breakdown\n\n")
        lines.append("| Category | Saved Locally | Target | Progress | Remaining |\n")
        lines.append("| :--- | :--- | :--- | :--- | :--- |\n")
        for cat, cst in ls["categories"].items():
            lines.append(f"| **{cat}** | **{cst['done']:,}** | {cst['total']:,} | {cst['pct']:.1f}% | {cst['remaining']:,} |\n")
        lines.append("\n")

    if SYNC_LOG_FILE.exists():
        lines.append("### Recent Sync Log\n```\n")
        try:
            with open(SYNC_LOG_FILE, "r") as f:
                for l in f.readlines()[-8:]:
                    lines.append(l)
        except Exception:
            pass
        lines.append("```\n\n")

    if LOG_FILE.exists():
        lines.append("### Recent Hourly Health Log\n```\n")
        try:
            with open(LOG_FILE, "r") as f:
                for l in f.readlines()[-10:]:
                    lines.append(l)
        except Exception:
            pass
        lines.append("```\n")

    with open(STATUS_MD, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LookMax Fleet Health, Sync & Auto-Destroy Controller")
    parser.add_argument("--cron", action="store_true",
                        help="Cron check: silent unless alert (exit 2) or healthy (exit 0)")
    parser.add_argument("--loop", action="store_true",
                        help="Continuous health monitoring loop (hourly by default)")
    parser.add_argument("--sync-loop", action="store_true",
                        help="Continuous 5-min pull+push zero-byte coordination sync daemon")
    parser.add_argument("--sync-once", action="store_true",
                        help="Run a single sync cycle (pull+push) then exit")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Interval seconds for --loop (default: 3600)")
    parser.add_argument("--sync-interval", type=int, default=SYNC_INTERVAL_SECS,
                        help=f"Interval seconds for --sync-loop (default: {SYNC_INTERVAL_SECS})")
    parser.add_argument("--set-api-key", type=str, help="Save Vast.ai API key securely")
    parser.add_argument("--test-api-key", action="store_true",
                        help="Validate Vast.ai API key and list active instances")
    parser.add_argument("--destroy-instance", type=int,
                        help="Manually destroy a specific Vast.ai instance ID")
    parser.add_argument("--local", action="store_true",
                        help="Print local Mac storage dataset summary (offline-safe, no SSH required)")
    args = parser.parse_args()

    if args.local:
        print_local_summary()
        return

    if args.set_api_key:
        set_vast_api_key(args.set_api_key)
        test_vast_api_key(args.set_api_key)
        return

    if args.test_api_key:
        test_vast_api_key()
        return

    if args.destroy_instance:
        key = get_vast_api_key()
        if not key:
            print("Error: No Vast.ai API key configured.")
            sys.exit(1)
        confirm = input(f"Destroy instance {args.destroy_instance}? (yes/no): ")
        if confirm.strip().lower() == "yes":
            ok = destroy_vast_instance(args.destroy_instance, key)
            print(f"{'✓ Destroyed' if ok else '✗ Failed to destroy'} instance {args.destroy_instance}.")
        return

    if args.sync_once:
        print("Running single sync cycle...")
        sync_markers(verbose=True)
        return

    if args.sync_loop:
        run_sync_loop(interval=args.sync_interval)
        return

    if args.loop:
        print(f"Starting fleet health monitor loop (interval: {args.interval}s)...")
        while True:
            code, msg = evaluate_fleet(cron_mode=False)
            if code == 2:
                print(f"[ALERT] {datetime.now()}: {msg}")
                try:
                    subprocess.run(["osascript", "-e",
                        'display notification "LookMax Fleet Alert!" with title "LookMax Alert"'],
                        check=False)
                except Exception:
                    pass
            time.sleep(args.interval)
    else:
        code, msg = evaluate_fleet(cron_mode=args.cron)
        sys.exit(code)


if __name__ == "__main__":
    main()
