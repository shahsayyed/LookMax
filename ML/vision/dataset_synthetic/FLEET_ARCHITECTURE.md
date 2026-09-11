# LookMax Synthetic Dataset — Fleet Architecture

**Status (2026-09-11): the 28,000-image generation run this fleet was built
for is complete, and every Vast.ai instance listed below has been
destroyed** — `fleet_monitor.py --test-api-key` confirms 0 active
instances. This document is kept as an architectural reference for how
multi-machine generation worked (relevant again if a future dataset
expansion needs the same approach), not as a description of anything
currently running. If `fleet_status.md` still shows a stale "unreachable"
alert, that's a leftover `crontab` entry still polling a destroyed host —
harmless, and fixed by removing the cron entry (see `PLAN.md`), not a sign
anything is actually wrong.

How the multi-machine Vast.ai generation fleet is provisioned, coordinated, and
monitored. Companion to [PLAN.md](PLAN.md), which covers the single-machine
generation pipeline itself (prompts, batching, output layout). This document
covers the layer on top: running that pipeline across several rented GPU boxes
at once, keeping their output in sync, and not paying for dead instances.

## Components

| File | Role |
| :--- | :--- |
| `fleet_monitor.py` | Single source of truth for the fleet. SSH health checks, hourly stall detection + auto-destroy, 30s marker sync daemon, Vast.ai REST API calls. |
| `monitor_fleet.sh` | Thin CLI wrapper around `fleet_monitor.py` (`sync`, `loop`, `status`, `set-api-key`, `destroy <id>`). |
| `fleet_state.json` | Generated. Per-host last-known count, stall timer, destroyed flag, pushed-marker set. |
| `fleet_status.md` | Generated. Human-readable status table, rewritten every time `evaluate_fleet()` runs. **Only as fresh as the last `--loop`/`--cron` run** — see Known Issues. |
| `fleet_monitor.log` / `fleet_sync.log` | Generated. Hourly health log / 30s sync log. |
| `add_machine.sh` | Registers a new Vast.ai box's SSH connection in `~/.ssh/config`. |
| `remote_deploy.sh` | rsyncs just the generator scripts (~616KB) to the box. |
| `setup_auto_resume.sh` | Writes `.auto_resume_config` on the box + installs an `onstart.sh` hook and a 5-min cron watchdog so a preempted spot instance self-resumes. |
| `run.sh` / `auto_resume.sh` | Runs on the remote box. Actual generation loop (tmux-managed `full_run.py`), dedupes against existing `.png` files. |
| `push_progress_to_new_machine.sh` | Disaster recovery: seed a brand-new box with completion markers for everything already generated, so it skips straight to the remaining work. |

## Task partitioning

The full dataset is 28,000 images, indexed 0–27999 by `full_run.py`'s
`build_full_task_list()`. Each host in `fleet_monitor.py`'s `HOSTS` dict owns a
contiguous, non-overlapping `slice` of that index range plus a Vast.ai
`instance_id` for billing control:

```python
HOSTS = {
    "vast1": {"slice": "4810:10000",  "target": 5190,  "instance_id": ...},
    "vast2": {"slice": "16000:28000", "target": 12000, "instance_id": ...},
    "vast3": {"slice": "13000:16000", "target": 3000,  "instance_id": ...},
    "vast4": {"slice": "10000:13000", "target": 3000,  "instance_id": ...},
}
```

`0:4810` is generated locally / already done. The five ranges must always sum
to 28,000 with no gaps or overlap — overlap means two machines burn GPU-hours
generating the same images.

## Provisioning a new machine

This is the exact sequence used to bring a box online, run from the Mac
unless noted:

```bash
# 1. Register SSH connection (paste the Vast.ai dashboard SSH string when prompted,
#    or pass IP/port directly). Writes/replaces the Host block for that alias
#    in ~/.ssh/config in place — reusing an alias name overwrites its old entry.
./add_machine.sh
```

```bash
# 2. Deploy the generator scripts (~616KB, not the whole repo) to the box.
./remote_deploy.sh <alias>
```

```bash
# 3. Install Python deps on the box.
ssh <alias> "cd /data/LookMax_Generator && ./run.sh setup"
```

```bash
# 4. Install auto-resume hooks (onstart.sh + 5-min cron watchdog) for this
#    machine's assigned slice, then start generation for the first time.
ssh <alias> "cd /data/LookMax_Generator && ./setup_auto_resume.sh <TASK_RANGE> <WORKERS>"
ssh <alias> "cd /data/LookMax_Generator && ./run.sh resume"
```

```bash
# 5. Add the host to fleet_monitor.py's HOSTS dict (slice, target, instance_id)
#    so it's covered by sync, health checks, and auto-destroy. load_state()
#    auto-detects a changed instance_id per host key and resets that host's
#    stall/marker state — no manual fleet_state.json editing needed.
```

```bash
# 6. Confirm it's alive and syncing.
python3 fleet_monitor.py --sync-once
```

`HF_TOKEN` does **not** need to be hardcoded anywhere — Qwen-Image-2512 is
Apache-2.0/ungated (see `install.sh`). `run.sh`, `auto_resume.sh`, and
`setup_auto_resume.sh` all read it from the calling shell's environment
(`${HF_TOKEN:-}`) and default to empty.

## Cross-machine sync (zero-byte marker coordination)

Every host generates only within its own slice, but images from *every* host
need to end up on the Mac, and every host needs to know what every *other*
host has already finished (in case a box gets replaced mid-slice and its
replacement needs to skip already-done work). `fleet_monitor.py --sync-loop`
runs this every 30s:

1. **Pull**: rsync real, non-empty, non-`.tmp` `.png` files from every
   reachable host into `ML/data/vision_synthetic/raw_generated/images/` on
   the Mac. `--ignore-existing` so a real local file is never clobbered.
2. **Push (delta only)**: for each reachable host, diff the Mac's real image
   set against that host's `pushed_markers` set (tracked per-host in
   `fleet_state.json`), touch zero-byte markers for only the *new* names into
   a scratch `.markers_delta_<host>/` dir, rsync just those over
   `--ignore-existing`, then record them as pushed. If there's nothing new,
   this is a no-op (0.00s) — this replaced an older design that rebuilt and
   pushed the *entire* marker set every cycle.

A host's own generation code (`full_run.py`) skips any filename that already
exists on disk, marker or real — that's what prevents duplicate generation
across machines.

**Gotcha:** because pushed markers land in the same directory the real
`.png`s are written to, a remote box's own `find *.png | wc -l` (which is what
`run.sh status` runs) counts markers too and will look far higher than that
box's actual generated count once sync has been running a while. Only
`fleet_monitor.py`'s remote check (`! -size 0`) and `--local` reports the true
count.

## Health monitoring & auto-destroy

`fleet_monitor.py --loop` (hourly) or `--cron` (for an external cron/launchd
job) calls `evaluate_fleet()`, which per host:

- SSHes in, reads image count (size-filtered), tmux session state, GPU
  utilization via `nvidia-smi`.
- If count increased since last check → resets the stall timer.
- If count is flat (or the host is unreachable) for **`STALL_ALERT_HOURS`**
  → destroys the Vast.ai instance via the REST API (if an API key is
  configured) and alerts, or just alerts if no key.
- On reaching its target count → runs a final verified rsync, then destroys
  the instance automatically so billing stops.

**`STALL_ALERT_HOURS` is currently `0.5`** (30 minutes) — dropped from an
earlier `3.0`. This is aggressive: a machine that's merely network-flaky (SSH
timing out without actually being dead) will get destroyed rather than just
flagged. Confirm this threshold is intentional given how often Vast.ai spot
boxes have transient connectivity blips.

**This loop only runs if something is actively invoking it.** `--sync-loop`
(the 30s marker daemon) does *not* do stall detection — it's a separate flag.
If neither `--loop` nor a cron `--cron` entry is running, stalls/deaths go
completely undetected and `fleet_status.md` silently goes stale while still
claiming the fleet is healthy.

## Known issues (found during this review)

- **No continuous health/auto-destroy loop was running** as of this review —
  only `--sync-loop`. Three of four registered hosts had already been
  terminated by Vast.ai (likely spot preemption) with nothing detecting or
  alerting on it. Run `fleet_monitor.py --loop &` (or wire up `--cron` via
  cron/launchd) to close this gap.
- **`run.sh status` overcounts** once sync has pushed markers to a box, per
  the Gotcha above — don't trust it for real progress once a machine has
  been part of a syncing fleet for a while; use `fleet_monitor.py --local` or
  the Mac-side `fleet_status.md` instead.
- **Alias reuse across setup sessions causes confusion.** `add_machine.sh`
  overwrites a `Host` block in place when you reuse an alias name, so
  `vast5` in `~/.ssh/config` today may point at a completely different
  physical machine than it did in an earlier session's notes. Cross-check the
  `HostName`/`Port` against `HOSTS` in `fleet_monitor.py`, not just the alias
  name, before trusting which box is which.
- **Orphaned host state isn't self-cleaning.** When a host is removed from
  `HOSTS` (e.g. after being replaced), its old entry stays in
  `fleet_state.json` forever and is never evaluated again — including never
  being checked for whether its instance still needs manual destruction on
  the Vast.ai console.
- **Never hardcode `HF_TOKEN` (or any secret) into `setup_auto_resume.sh` or
  any other tracked script.** A real token was found hardcoded here during
  this review and has been reverted to reading from the environment — it
  wasn't functionally needed (the model repo is ungated) and would have been
  committed to git history permanently.
