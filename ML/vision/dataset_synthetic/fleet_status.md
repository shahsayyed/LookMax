# LookMax Synthetic Dataset Generation -- Fleet Status

*Last Checked: 2026-09-08 01:00:03* | *Vast.ai API Key: Configured (Auto-Destroy Active)*

> [!NOTE]
> Fleet is healthy and actively generating synthetic data.

| Host | Instance ID | Slice Target | Completed | Progress | GPUs | Stalled Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vast1** | `50209314` | 4810:10000 (5190) | **557** | 10.7% | 100%/100% | 0h (active) | 🟢 RUNNING |
| **vast2** | `50210080` | 16000:28000 (12000) | **485** | 4.0% | 100%/0% | 0h (active) | 🟢 RUNNING |
| **vast3** | `50199060` | 13000:16000 (3000) | **813** | 27.1% | 100% | 0h (active) | 🟢 RUNNING |
| **vast4** | `50197596` | 10000:13000 (3000) | **701** | 23.4% | 100% | 0h (active) | 🟢 RUNNING |
| **TOTAL** | - | **23190** | **2556** | **11.0%** | - | - | - |

### Local Storage Progress (Offline-Safe)

**Total Real Images Saved Locally on Mac**: `18,022` / `28,000` (**64.4%** complete)

| Slice Range | Target | Saved Locally | Progress | Remaining |
| :--- | :--- | :--- | :--- | :--- |
| **[0:4810] (Original)** | 4,810 | **4,734** | 98.4% | 76 |
| **[4810:10000] (vast1 slice)** | 5,190 | **2,650** | 51.1% | 2,540 |
| **[10000:13000] (vast4 slice)** | 3,000 | **1,229** | 41.0% | 1,771 |
| **[13000:16000] (vast3 slice)** | 3,000 | **1,704** | 56.8% | 1,296 |
| **[16000:28000] (vast2 slice)** | 12,000 | **7,705** | 64.2% | 4,295 |

#### Local Category Breakdown

| Category | Saved Locally | Target | Progress | Remaining |
| :--- | :--- | :--- | :--- | :--- |
| **Men_Grooming** | **3,790** | 6,000 | 63.2% | 2,210 |
| **Women_Grooming** | **3,843** | 6,000 | 64.0% | 2,157 |
| **Men_Outfit** | **5,176** | 8,000 | 64.7% | 2,824 |
| **Women_Outfit** | **5,213** | 8,000 | 65.2% | 2,787 |

### Recent Sync Log
```
[2026-09-08 00:59:21] PUSH markers->vast1: ok (19 new delta markers sent to remote)
[2026-09-08 00:59:21] PUSH markers->vast2: ok (19 new delta markers sent to remote)
[2026-09-08 00:59:23] PUSH markers->vast3: ok (19 new delta markers sent to remote)
[2026-09-08 00:59:25] PUSH markers->vast4: ok (19 new delta markers sent to remote)
[2026-09-08 00:59:26] Cycle done. Local total=18004. Reachable=['vast1', 'vast2', 'vast3', 'vast4']
[2026-09-08 00:59:58] PULL vast1: +7 new
[2026-09-08 01:00:00] PULL vast2: +7 new
[2026-09-08 01:00:04] PULL vast3: +4 new
```

### Recent Hourly Health Log
```
[2026-09-07 22:48:23] | vast1: 477/5190 (9.2%) [RUNNING] GPUs: 100% (Δ+257, stalled=0.0h) | vast2: 972/9000 (10.8%) [RUNNING] GPUs: 100%/100% (Δ+353, stalled=0.0h) | vast3: 268/3000 (8.9%) [RUNNING] GPUs: 100% (Δ+255, stalled=0.0h) | vast4: 274/3000 (9.1%) [RUNNING] GPUs: 100% (Δ+195, stalled=0.0h) | vast5: 107/3000 (3.6%) [RUNNING] GPUs: 100% (Δ+107, stalled=0.0h)
[2026-09-07 22:55:03] | vast1: 505/5190 (9.7%) [RUNNING] GPUs: 100% (Δ+285, stalled=0.0h) | vast2: 0/9000 (0.0%) [RUNNING] GPUs: 0% (Δ+0, stalled=0.0h) | vast3: 295/3000 (9.8%) [RUNNING] GPUs: 100% (Δ+282, stalled=0.0h) | vast4: 295/3000 (9.8%) [RUNNING] GPUs: 100% (Δ+216, stalled=0.0h) | vast5: 122/3000 (4.1%) [RUNNING] GPUs: 100% (Δ+122, stalled=0.0h)
[2026-09-07 22:58:12] | vast1: 518/5190 (10.0%) [RUNNING] GPUs: 100% (Δ+13, stalled=0.0h) | vast2: 0/9000 (0.0%) [RUNNING] GPUs: 0% (Δ+0, stalled=0.1h) | vast3: 308/3000 (10.3%) [RUNNING] GPUs: 100% (Δ+13, stalled=0.0h) | vast4: 306/3000 (10.2%) [RUNNING] GPUs: 100% (Δ+11, stalled=0.0h) | vast5: 130/3000 (4.3%) [RUNNING] GPUs: 100% (Δ+8, stalled=0.0h)
[2026-09-07 23:03:35] | vast1: 541/5190 (10.4%) [RUNNING] GPUs: 100% (Δ+23, stalled=0.0h) | vast2: 0/9000 (0.0%) [RUNNING] GPUs: 0% (Δ+0, stalled=0.1h) | vast3: 329/3000 (11.0%) [RUNNING] GPUs: 76% (Δ+21, stalled=0.0h) | vast4: 324/3000 (10.8%) [RUNNING] GPUs: 100% (Δ+18, stalled=0.0h) | vast5: 0/3000 (0.0%) [RUNNING] GPUs: 100% (Δ+0, stalled=0.0h)
[2026-09-07 23:12:47] | vast1: 0/5190 (0.0%) [RUNNING] GPUs: 0% (Δ+0, stalled=0.0h) | vast2: 7/9000 (0.1%) [RUNNING] GPUs: 100% (Δ+7, stalled=0.0h) | vast3: 368/3000 (12.3%) [RUNNING] GPUs: 100% (Δ+60, stalled=0.0h) | vast4: 355/3000 (11.8%) [RUNNING] GPUs: 100% (Δ+49, stalled=0.0h) | vast5: UNREACHABLE (SSH failed: ssh: connect to host 167.172.47.205 port) [Stalled: 0.0h]
[2026-09-07 23:14:02] | vast1: 2/5190 (0.0%) [RUNNING] GPUs: 100% (Δ+2, stalled=0.0h) | vast2: 10/9000 (0.1%) [RUNNING] GPUs: 100% (Δ+10, stalled=0.0h) | vast3: 374/3000 (12.5%) [RUNNING] GPUs: 100% (Δ+66, stalled=0.0h) | vast4: 358/3000 (11.9%) [RUNNING] GPUs: 100% (Δ+52, stalled=0.0h) | vast5: 0/3000 (0.0%) [STOPPED] GPUs: 0% (Δ+0, stalled=0.0h)
[2026-09-07 23:17:19] | vast1: 14/5190 (0.3%) [RUNNING] GPUs: 100% (Δ+14, stalled=0.0h) | vast2: 17/9000 (0.2%) [RUNNING] GPUs: 99% (Δ+17, stalled=0.0h) | vast3: 388/3000 (12.9%) [RUNNING] GPUs: 100% (Δ+80, stalled=0.0h) | vast4: 369/3000 (12.3%) [RUNNING] GPUs: 100% (Δ+63, stalled=0.0h) | vast5: 0/3000 (0.0%) [RUNNING] GPUs: 0% (Δ+0, stalled=0.0h)
[2026-09-08 00:06:06] | vast1: 142/5190 (2.7%) [RUNNING] GPUs: 100%/100% (Δ+142, stalled=0.0h) | vast2: 66/12000 (0.5%) [RUNNING] GPUs: 100%/100% (Δ+66, stalled=0.0h) | vast3: 587/3000 (19.6%) [RUNNING] GPUs: 100% (Δ+199, stalled=0.0h) | vast4: 527/3000 (17.6%) [RUNNING] GPUs: 100% (Δ+158, stalled=0.0h)
[2026-09-08 00:30:04] | vast1: 327/5190 (6.3%) [RUNNING] GPUs: 0%/100% (Δ+185, stalled=0.0h) | vast2: 253/12000 (2.1%) [RUNNING] GPUs: 100%/100% (Δ+187, stalled=0.0h) | vast3: 686/3000 (22.9%) [RUNNING] GPUs: 0% (Δ+99, stalled=0.0h) | vast4: 602/3000 (20.1%) [RUNNING] GPUs: 100% (Δ+75, stalled=0.0h)
[2026-09-08 01:00:03] | vast1: 557/5190 (10.7%) [RUNNING] GPUs: 100%/100% (Δ+415, stalled=0.0h) | vast2: 485/12000 (4.0%) [RUNNING] GPUs: 100%/0% (Δ+419, stalled=0.0h) | vast3: 813/3000 (27.1%) [RUNNING] GPUs: 100% (Δ+226, stalled=0.0h) | vast4: 701/3000 (23.4%) [RUNNING] GPUs: 100% (Δ+174, stalled=0.0h)
```
