# LookMax Synthetic Dataset Generation -- Fleet Status

*Last Checked: 2026-09-07 15:27:15* | *Vast.ai API Key: Configured (Auto-Destroy Active)*

> [!CAUTION]
> 🚨 vast2 has been UNREACHABLE for 6.5 hours! Error: ssh: connect to host 93.91.156.94 port 40891: Connection refused

| Host | Instance ID | Slice Target | Completed | Progress | GPUs | Stalled Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vast2** | `50095177` | 16000:28000 (12000) | **4893** | 0.0% | N/A | 6.5h | 🔴 UNREACHABLE |
| **vast1** | `50166753` | 4810:13000 (8190) | **0** | 0.0% | 0%/0%/0%/0% | 0h (active) | 🟢 RUNNING |
| **vast3** | `50149174` | 13000:16000 (3000) | **137** | 4.6% | 0% | 0h (active) | 🟢 RUNNING |
| **TOTAL** | - | **23190** | **5030** | **21.7%** | - | - | - |

### Recent Sync Log
```
[2026-09-07 15:26:26] PUSH markers->vast3: ok (0 new markers sent to remote, 12409 total)
[2026-09-07 15:26:26] Cycle done. Local total=12409. Reachable=['vast1', 'vast3']
[2026-09-07 15:26:56] PULL vast2: unreachable — skipped
[2026-09-07 15:26:58] PULL vast1: +0 new
[2026-09-07 15:26:59] PULL vast3: +0 new
[2026-09-07 15:27:00] PUSH markers->vast1: ok (0 new markers sent to remote, 12409 total)
[2026-09-07 15:27:02] PUSH markers->vast3: ok (0 new markers sent to remote, 12409 total)
[2026-09-07 15:27:02] Cycle done. Local total=12409. Reachable=['vast1', 'vast3']
```

### Recent Hourly Health Log
```
[2026-09-07 00:00:02] | vast2: 1152/12000 (9.6%) [RUNNING] GPUs: 100%/100% (Δ+286, stalled=0.0h) | vast3: 464/11190 (4.1%) [RUNNING] GPUs: 100%/0% (Δ+202, stalled=0.0h)
[2026-09-07 01:00:07] | vast2: 1608/12000 (13.4%) [RUNNING] GPUs: 100%/100% (Δ+456, stalled=0.0h) | vast3: UNREACHABLE (SSH failed: ssh: connect to host 117.18.102.50 port ) [Stalled: 0.0h]
[2026-09-07 06:00:08] | vast1: UNREACHABLE (SSH failed: ssh: connect to host 162.43.139.39 port ) [Stalled: 0.0h] | vast2: 3966/12000 (33.1%) [RUNNING] GPUs: 100%/100% (Δ+2358, stalled=0.0h) | vast3: UNREACHABLE (SSH failed: ssh: connect to host 117.18.102.50 port ) [Stalled: 5.0h]
[2026-09-07 07:43:32] | vast1: 0/11190 (0.0%) [RUNNING] GPUs: 0%/0% (Δ+0, stalled=1.7h) | vast2: 4789/12000 (39.9%) [RUNNING] GPUs: 100%/100% (Δ+823, stalled=0.0h)
[2026-09-07 07:44:48] | vast1: 2/11190 (0.0%) [RUNNING] GPUs: 100%/100% (Δ+2, stalled=0.0h) | vast2: 4797/12000 (40.0%) [RUNNING] GPUs: 100%/99% (Δ+8, stalled=0.0h)
[2026-09-07 07:56:40] | vast2: 4891/12000 (40.8%) [RUNNING] GPUs: 100%/100% (Δ+94, stalled=0.0h) | vast1: 93/11190 (0.8%) [RUNNING] GPUs: 100%/100% (Δ+-10629, stalled=0.0h)
[2026-09-07 09:00:07] | vast2: UNREACHABLE (SSH failed: ssh: connect to host 93.91.156.94 port 4) [Stalled: 0.0h] | vast1: 188/11190 (1.7%) [RUNNING] GPUs: 100%/100% (Δ+-10541, stalled=1.1h)
[2026-09-07 11:00:07] | vast2: UNREACHABLE (SSH failed: ssh: connect to host 93.91.156.94 port 4) [Stalled: 2.0h] | vast1: UNREACHABLE (SSH failed: ssh: connect to host 93.91.156.94 port 4) [Stalled: 3.1h]
[2026-09-07 11:58:43] | vast2: UNREACHABLE (SSH failed: ssh: connect to host 93.91.156.94 port 4) [Stalled: 3.0h] | vast1: UNREACHABLE (SSH failed: ssh: connect to host 93.91.156.94 port 4) [Stalled: 4.0h] | vast3: 0/3000 (0.0%) [RUNNING] GPUs: 0% (Δ+0, stalled=0.0h)
[2026-09-07 15:27:15] | vast2: UNREACHABLE (SSH failed: ssh: connect to host 93.91.156.94 port 4) [Stalled: 6.5h] | vast1: 0/8190 (0.0%) [RUNNING] GPUs: 0%/0%/0%/0% (Δ+0, stalled=0.0h) | vast3: 137/3000 (4.6%) [STOPPED] GPUs: 0% (Δ+137, stalled=0.0h)
```
