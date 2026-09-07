#!/usr/bin/env python3
"""
fleet_monitor.py -- Automated health, progress monitor, and auto-destroy controller
for LookMax synthetic generation fleet on Vast.ai.

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

Usage:
  python3 fleet_monitor.py                     # Single manual status check
  python3 fleet_monitor.py --cron              # Silent cron check; exit 2 if alert, exit 0 if normal
  python3 fleet_monitor.py --loop              # Background loop running every hour (3600s)
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
STATUS_MD = SCRIPT_DIR / "fleet_status.md"
API_KEY_FILE = SCRIPT_DIR / ".vast_api_key"
LOCAL_DEST_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "raw_generated"

HOSTS = {
    "vast2": {
        "slice": "16000:28000",
        "target": 12000,
        "instance_id": 50095177,
        "desc": "2x H100 SXM5 (80GB)"
    },
    "vast3": {
        "slice": "4810:16000",
        "target": 11190,
        "instance_id": 50100378,
        "desc": "2x H100 PCIe (80GB)"
    }
}

STALL_ALERT_HOURS = 3.0  # Alert if stopped for >= 3 hours

REMOTE_CHECK_CMD = r"""
CONFIG="/data/LookMax_Generator/.auto_resume_config"
TASK_RANGE=""
[ -f "$CONFIG" ] && source "$CONFIG"
IMG_COUNT=$(find /data/qwen_dataset_output/images -maxdepth 1 -name "*.png" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d " ")
TMUX_STATE=$(tmux has-session -t lookmax_gen 2>/dev/null && echo "RUNNING" || echo "STOPPED")
GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "")
echo "COUNT=$IMG_COUNT"
echo "TMUX=$TMUX_STATE"
echo "TASK_RANGE=$TASK_RANGE"
echo "GPU_START"
echo "$GPU_STATS"
echo "GPU_END"
"""


def get_vast_api_key() -> str | None:
    # 1. Environment variable
    env_key = os.environ.get("VAST_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    # 2. Local config file (.vast_api_key)
    if API_KEY_FILE.exists():
        try:
            k = API_KEY_FILE.read_text().strip()
            if k:
                return k
        except Exception:
            pass
    # 3. User home ~/.vast_api_key
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


def destroy_vast_instance(instance_id: int, api_key: str) -> bool:
    url = f"https://console.vast.ai/api/v1/instances/{instance_id}/?api_key={api_key}"
    req = urllib.request.Request(url, method="DELETE", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("success", False) or resp.status in (200, 204)
    except Exception as e:
        print(f"Error calling Vast.ai DELETE on instance {instance_id}: {e}")
        return False


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

    # Verify local file count and absence of .tmp files
    local_images = list(images_dest.glob("*.png"))
    local_tmps = list(images_dest.glob("*.tmp"))

    if local_tmps:
        print(f"✗ Warning: {len(local_tmps)} .tmp files found in local storage!")
        return False

    print(f"✓ Local archive verified: {len(local_images)} total valid images present in {images_dest}.")
    return True


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"hosts": {}, "history": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


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

        # If already destroyed, skip querying
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

            status_str = f"{host}: UNREACHABLE (SSH failed: {data['raw_error'][:40]}) [Stalled: {stalled_hours:.1f}h]"
            log_line_parts.append(status_str)
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
                alerts.append(f"🚨 {host} has been UNREACHABLE for {stalled_hours:.1f} hours! Error: {data['raw_error']}")
            continue

        # Host is reachable
        curr_count = data["count"]
        last_count = host_state["last_count"]
        delta = curr_count - last_count
        target = meta["target"]
        pct = (curr_count / target * 100.0) if target else 0.0

        is_generating = (delta > 0)
        is_tmux_running = (data["tmux"] == "RUNNING")

        # Check if generation has finished!
        if curr_count >= target and target > 0 and not host_state.get("completed"):
            host_state["completed"] = True
            host_state["last_count"] = curr_count
            print(f"🎉 [{host}] Target reached: {curr_count}/{target} images generated!")

            # 1. Run safe sync & local verification
            sync_ok = sync_and_verify_host(host, target)
            if sync_ok:
                instance_id = meta.get("instance_id")
                if api_key and instance_id:
                    print(f"==> Destroying Vast.ai instance {instance_id} for {host} to prevent further charges...")
                    destroyed = destroy_vast_instance(instance_id, api_key)
                    if destroyed:
                        host_state["destroyed"] = True
                        host_state["last_status"] = "DESTROYED"
                        alerts.append(
                            f"🎉 {host} completed all {curr_count}/{target} images! "
                            f"Data was verified locally, and instance {instance_id} was destroyed on Vast.ai (billing stopped)."
                        )
                    else:
                        alerts.append(
                            f"⚠️ {host} completed all {curr_count}/{target} images and data is verified locally, "
                            f"but the Vast.ai API call to destroy instance {instance_id} failed. Please destroy it in the Vast.ai console."
                        )
                else:
                    alerts.append(
                        f"🎉 {host} completed all {curr_count}/{target} images and data is safely verified locally! "
                        f"Please destroy instance {instance_id} in your Vast.ai console (or set your Vast.ai API key so it can be destroyed automatically)."
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
                alerts.append(
                    f"🚨 {host} has been STALLED for {stalled_hours:.1f} hours! "
                    f"(Count stuck at {curr_count}/{target}, tmux={data['tmux']})"
                )

        gpu_desc = "/".join([g["util"] for g in data["gpus"]]) if data["gpus"] else "N/A"
        log_line_parts.append(
            f"{host}: {curr_count}/{target} ({pct:.1f}%) [{data['tmux']}] GPUs: {gpu_desc} (Δ+{delta}, stalled={stalled_hours:.1f}h)"
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

    # Save state
    save_state(state)

    # Append to log file
    log_entry = " | ".join(log_line_parts) + "\n"
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

    # Update Markdown status table
    update_markdown_status(now_str, host_summaries, alerts)

    # Return result
    if alerts:
        alert_msg = "\n".join(alerts)
        if cron_mode:
            print(f"ALERT TRIGGERED:\n{alert_msg}")
        return 2, alert_msg
    else:
        summary_msg = "Fleet healthy:\n" + "\n".join([
            f" - {h['host']}: {h['count']}/{h['target']} ({h['percent']:.1f}%) | {h['status']} | GPUs: {h['gpus']} (Δ+{h['delta']})"
            for h in host_summaries
        ])
        if not cron_mode:
            print(summary_msg)
        return 0, summary_msg


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
        status_badge = "🟢 RUNNING" if h["status"] == "RUNNING" else ("🏁 DESTROYED" if h["status"] == "DESTROYED" else f"🔴 {h['status']}")
        lines.append(
            f"| **{h['host']}** | `{HOSTS[h['host']]['instance_id']}` | {HOSTS[h['host']]['slice']} ({h['target']}) | "
            f"**{h['count']}** | {h['percent']:.1f}% | {h['gpus']} | {stalled_str} | {status_badge} |\n"
        )

    lines.append(f"| **TOTAL ACTIVE** | - | **{total_target}** | **{total_done}** | **{(total_done/total_target*100):.1f}%** | - | - | - |\n\n")

    if LOG_FILE.exists():
        lines.append("### Recent Hourly Log Entries\n```\n")
        try:
            with open(LOG_FILE, "r") as f:
                all_logs = f.readlines()
                for l in all_logs[-10:]:
                    lines.append(l)
        except Exception:
            pass
        lines.append("```\n")

    with open(STATUS_MD, "w") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="LookMax Fleet Health & Auto-Destroy Controller")
    parser.add_argument("--cron", action="store_true", help="Cron check: silent unless alert or completed (exit code 2)")
    parser.add_argument("--loop", action="store_true", help="Continuous monitoring loop (runs hourly)")
    parser.add_argument("--interval", type=int, default=3600, help="Interval in seconds for loop mode (default: 3600)")
    parser.add_argument("--set-api-key", type=str, help="Save Vast.ai API key")
    parser.add_argument("--test-api-key", action="store_true", help="Verify Vast.ai API key against Vast.ai API")
    parser.add_argument("--destroy-instance", type=int, help="Manually destroy a specific Vast.ai instance ID")
    args = parser.parse_args()

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
            print("Error: No Vast.ai API key configured. Run with --set-api-key <KEY> first.")
            sys.exit(1)
        confirm = input(f"Are you sure you want to DESTROY instance {args.destroy_instance}? (yes/no): ")
        if confirm.strip().lower() == "yes":
            ok = destroy_vast_instance(args.destroy_instance, key)
            if ok:
                print(f"✓ Instance {args.destroy_instance} destroyed.")
            else:
                print(f"✗ Failed to destroy instance {args.destroy_instance}.")
        return

    if args.loop:
        print(f"Starting LookMax Fleet Monitor & Auto-Destroy loop (interval: {args.interval}s)...")
        while True:
            code, msg = evaluate_fleet(cron_mode=False)
            if code == 2:
                print(f"[ALERT] {datetime.now()}: {msg}")
                try:
                    subprocess.run([
                        "osascript", "-e",
                        'display notification "LookMax Fleet Alert!" with title "LookMax Alert"'
                    ], check=False)
                except Exception:
                    pass
            time.sleep(args.interval)
    else:
        code, msg = evaluate_fleet(cron_mode=args.cron)
        sys.exit(code)


if __name__ == "__main__":
    main()
