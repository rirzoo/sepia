# Benchmark

How Sepia decides which device runs each pipeline stage, and the numbers so far. The results are also the project's secondary output ([OVERVIEW.md](OVERVIEW.md#goal)).

## Harness

The benchmark is a mode of sepiad itself, so the numbers describe the pipeline users actually run:

```bash
sepiad --infer NPU --post cpu --bench 60   # run 60 s, append one row to bench.csv
```

**Baselines** (without these the numbers mean nothing):

1. Idle, camera off
2. Camera streaming, no effects (the cost of capture and MJPEG decode alone)
3. Meet in Chrome, real camera, Meet's blur on (the bar to beat), compared with Meet using Sepia Cam with Meet's blur off

**Matrix:** inference on {CPU, GPU, NPU} × blur and blend on {CPU, iGPU}: 6 configurations.

**Meet comparison:** `scripts/bench_meet.py` runs in a terminal during a Meet call. Before each phase it says how to set Meet and waits for Enter, then measures the whole laptop for 60 s. Phases can be picked by name; `meet+none` reads Sepia Cam with sepiad as a passthrough (`--infer none`).

## Metrics and how they're measured on this laptop

| Metric | Source | Root? |
| --- | --- | --- |
| Chip power by domain | RAPL energy counters in `/sys/class/powercap`: package, CPU cores, iGPU (`uncore`) and platform (`psys`), read at the start and end of each phase. The NPU has no domain of its own; its power is in package minus cores minus iGPU | Yes, or open them until reboot ([SETUP.md](SETUP.md), step 6) |
| Whole-system power | `/sys/class/power_supply/BAT1/power_now`, sampled at 1 Hz, on battery. Cross-check only: idle readings swung between 5.4 and 11.0 W, too noisy for differences of a few watts | No |
| NPU utilisation | Change in `/sys/class/accel/accel0/device/npu_busy_time_us` over the run | No |
| Time per stage | `time.perf_counter()` around each step | No |
| fps and latency (mean, p95) | Capture and send timestamps | No |
| sepiad's added delay (mean, p95) | Latency minus one frame period (33.3 ms), the camera's own transfer ([ARCHITECTURE.md](ARCHITECTURE.md#latency-and-lip-sync)) | No |
| sepiad CPU % | `/proc/self/stat` | No |

The NPU-utilisation counter also proves the NPU is doing the inference when it is selected.

## Method

On battery, or plugged in with the CPU on the battery profile (below). Screen brightness and Wi-Fi stay as they are, and no new work starts during the run. Steady background apps can stay open. Run the matrix under `systemd-inhibit` (see `scripts/bench_matrix.py`): on Oct 22, 2026 the laptop suspended on its own during a run. Each configuration runs for 60 s, and idle phases alternate with the configurations; each configuration is compared with the mean of the idle phases on either side of it, so slow changes in background load cancel out. Each idle phase starts after a 15 s cooldown, because power stays high for several seconds after a load ends. The spread of the idle phases is the noise floor, and smaller differences don't count. The whole sequence runs 3 times, and results are reported as mean ± standard deviation. Raw numbers and the method are published with the README.

**Plugged in.** On battery, Fedora's tuned switches to the `balanced-battery` profile. Its only CPU setting is the energy-performance preference, `balance_power`, which keeps clocks low. Setting that profile by hand while plugged in ([SETUP.md](SETUP.md), step 7) gives the same CPU behaviour, so runs don't drain the battery and can go overnight. On Oct 22, 2026 a plugged-in run agreed with the battery runs within their spread ([Mask from the previous frame](#mask-from-the-previous-frame-plugged-in-oct-22-2026)). Every bench.csv row records the preference in its `epp` column, and `bench_matrix.py` warns when it isn't `balance_power`. Below 20% battery GNOME switches to power saver (`epp` is `power`), which is slower still, so those runs don't count. Plugged in, `psys` includes the charger; compare package power only.

**Latency only.** Comparisons of latency and fps skip the idle phases: `sepiad --bench 60` 3 times per variant, alternating, about 3 minutes per variant.

## Results

### Model only (Oct 21, 2026)

`selfie_multiclass_256x256.tflite`, 200 inferences per device on random input, OpenVINO 2025.1, `intel-npu-driver` 1.16.0 (power state not recorded):

| Device | Compile | Inference per frame | Notes |
| --- | --- | --- | --- |
| NPU | 2.3 s | 8.7 ms | NPU busy time rose 1636 ms over the 200 runs, so the NPU did the work |
| iGPU | 1.5 s | 7.2 ms | |
| CPU | 0.2 s | 52 ms | Too slow for 30 fps |

All three gave identical per-pixel classes (on random input; to be repeated on real frames). The full pipeline, power and baselines are still to run.

### Model-only power (Oct 22, 2026)

`scripts/bench_infer_power.py`, committed with this entry. `selfie_multiclass_256x256.tflite` at 30 inferences per second, random input, camera off, 60 s per phase, 3 runs starting at 11:08, 11:18 and 11:26 IST. OpenVINO 2025.1, `intel-npu-driver` 1.16.0, on battery, with Chrome and VS Code open in the background. Idle package power was between 2.86 and 3.34 W, and the idle spread within each run was 0.23, 0.48 and 0.09 W.

Power added over idle, mean ± standard deviation of the 3 runs:

| Device | Package | CPU cores | iGPU | Rest of chip (includes the NPU) | Energy per inference | Inferences/s | CPU use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NPU | +1.95 ± 0.14 W | +0.73 W | 0.00 W | +1.21 W | 64.9 ± 4.5 mJ | 30.0 | 4% |
| iGPU | +5.64 ± 0.10 W | +0.66 W | +4.15 W | +0.84 W | 188.1 ± 3.4 mJ | 30.0 | 32% |
| CPU | +13.76 ± 0.16 W | +13.12 W | 0.00 W | +0.65 W | 1001.8 ± 9.9 mJ | 13.8 | 609% |

NPU busy time was 39% of wall time in every NPU phase, so the NPU ran the inference. Battery power is left out because it was too noisy (see Metrics).

**Decision:** inference runs on the NPU by default. At the same 30 inferences per second it adds about a third of the iGPU's package power (1.95 vs 5.64 W), and it leaves the iGPU idle for blur and blend. The CPU can't reach 30 inferences per second, and it draws about 7 times the NPU's power for 15 times the energy per inference. This covers the model alone; the full pipeline benchmark still has to confirm it ([ARCHITECTURE.md](ARCHITECTURE.md#inference-on-the-npu)).

### First blur pipeline timing (Oct 22, 2026)

`sepiad --bench` at 17:42 to 17:43 IST, from a first draft of sepiad that ran the preprocessing and the softmax on the CPU. Real camera at 720p30 MJPG, output to `/dev/video10` with nothing reading it, OpenVINO 2025.1, `intel-npu-driver` 1.16.0. Plugged in with the battery full, and the RAPL counters were still closed after a reboot, so there are no power numbers. One run per configuration, 20 s each after 30 warm-up frames, so treat it as a first look.

| Configuration | fps | Latency mean / p95 | sepiad CPU | NPU busy | Decode | Preprocess | Inference | Blur and blend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `--infer NPU --post cpu` | 29.8 | 60.0 / 63.4 ms | 109% | 38% | 3.69 ms | 0.89 ms | 18.0 ms | 4.49 ms |
| `--infer NPU --post gpu` | 29.8 | 59.0 / 62.4 ms | 37% | 38% | 3.82 ms | 0.34 ms | 17.7 ms | 4.2 ms |
| `--infer none` (no effects) | 29.8 | 37.7 / 41.3 ms | 16% | 0% | 4.82 ms | 0 | 0 | 0 |

In these three runs the inference column also includes the softmax. A fourth run of `--infer NPU --post cpu` (15 s) timed it on its own: inference 13.95 ms, softmax 4.0 ms, blur and blend 4.45 ms, 29.8 fps, latency 59.4 / 63.7 ms, CPU 105%. Sending to the virtual camera took about 0.5 ms in every run.

Every configuration held the camera's frame rate. The blur pipeline adds about 22 ms of latency over the no-effects baseline, well inside the 66 ms budget. Inference took 14 ms in the loop, against 8.7 ms on random input with nothing else running; the cause isn't known yet. Moving blur and blend to the iGPU cut sepiad's CPU use from 109% to 37% at about the same stage time. Whether that also saves power is for the matrix to show.

### Blur pipeline power matrix (Oct 22, 2026)

`scripts/bench_matrix.py`, committed with this entry: `sepiad --bench 60` for each configuration, with a 60 s idle phase before and after each one, run 3 times. Real camera at 720p30 MJPG, output to `/dev/video10` with nothing reading it, OpenVINO 2025.1, `intel-npu-driver` 1.16.0. On battery the whole time (79% at the start, 45% at the end), power profile `balance_power`, with Chrome and VS Code open in the background. Raw rows for every phase, idle ones included, are in `bench.csv`.

The matrix ran twice. The first pass (17:51 to 18:43 IST) used the same draft as the entry above, with the preprocessing and the softmax on the CPU. That draft dropped to about 22 fps on battery, so the preprocessing and the softmax moved into the model, where they run on the inference device. The second pass (18:47 to 19:38 IST) used that version, which is the committed one. Idle package power was between 2.55 and 2.95 W in the first pass and between 2.12 and 2.88 W in the second.

Second pass (committed code), mean ± standard deviation of the 3 runs, power added over idle:

| Inference | Blur and blend | Package | CPU cores | iGPU | Rest of chip (includes the NPU) | fps | Latency mean / p95 | sepiad CPU | NPU busy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none (camera only) | – | +2.42 ± 0.32 W | +2.10 W | 0.00 W | +0.32 W | 25.8 | 128.8 / 221.2 ms | 74% | 0% |
| NPU | CPU | +6.19 ± 0.14 W | +4.01 W | 0.00 W | +2.18 W | 29.0 | 104.1 / 180.7 ms | 128% | 39% |
| NPU | iGPU | +6.55 ± 0.25 W | +4.14 W | +0.22 W | +2.18 W | 29.1 | 94.8 / 173.1 ms | 51% | 39% |
| GPU | CPU | +10.12 ± 0.19 W | +4.66 W | +4.17 W | +1.29 W | 29.8 | 59.8 / 85.4 ms | 139% | 0% |
| GPU | iGPU | +10.26 ± 0.20 W | +4.46 W | +4.54 W | +1.25 W | 29.6 | 59.0 / 86.2 ms | 68% | 0% |
| CPU | CPU | +12.83 ± 0.33 W | +11.94 W | 0.00 W | +0.89 W | 11.5 | 415.8 / 567.0 ms | 576% | 0% |
| CPU | iGPU | +12.19 ± 0.34 W | +11.20 W | +0.10 W | +0.89 W | 11.0 | 438.7 / 497.1 ms | 550% | 0% |

First pass (draft, preprocessing and softmax on the CPU), package power over idle and fps:

| Inference | Blur and blend | Package | fps | Latency mean | sepiad CPU |
| --- | --- | --- | --- | --- | --- |
| none (camera only) | – | +2.33 ± 0.26 W | 26.3 | 122.2 ms | 72% |
| NPU | CPU | +5.58 ± 0.20 W | 22.3 | 207.6 ms | 124% |
| NPU | iGPU | +6.19 ± 0.29 W | 23.1 | 199.4 ms | 57% |
| GPU | CPU | +10.91 ± 0.19 W | 29.7 | 78.7 ms | 171% |
| GPU | iGPU | +11.41 ± 0.27 W | 29.8 | 63.1 ms | 87% |
| CPU | CPU | +12.10 ± 0.23 W | 11.4 | 423.1 ms | 555% |
| CPU | iGPU | +11.35 ± 0.08 W | 10.4 | 462.3 ms | 526% |

On battery the CPU runs at low clocks, and the draft's CPU-side work no longer fitted in a frame: decode took about 8.3 ms and the softmax about 9.4 ms, against 3.7 and 4.0 ms when plugged in. Moving the softmax and preprocessing onto the device brought the NPU from 22.3 to 29.0 fps and halved its latency. Package power rose from +5.58 to +6.19 W because it now handles about 30% more frames; per frame it fell from about 250 to 213 mJ. GPU inference got cheaper and faster with the same change (+10.91 to +10.12 W, 78.7 to 59.8 ms).

The camera alone can't hold 30 fps on battery either. With no effects, decode took 24 to 31 ms per frame, because the light load leaves the CPU at its lowest clocks, and the frame rate fell to about 26 fps with 129 ms of latency. Under the NPU pipeline's heavier load decode took 10.9 to 12.1 ms, and inference 14.6 to 15.0 ms; the stages add up to just over the 33 ms frame time, so NPU runs reach 28.6 to 29.5 fps.

**Decision:** inference stays on the NPU. NPU inference adds 6.19 W against 10.12 W for the iGPU, about 40% less, at almost the same frame rate. CPU inference is ruled out again: under 12 fps at the highest power of all.

Where blur and blend run is a near tie, and the choice is provisional. On the iGPU they cut sepiad's CPU use from 128% to 51% and added 0.36 W of package power. That gap is close to the noise: the two configurations' standard deviations were 0.14 and 0.25 W over 3 runs, and the idle phases of the pass spanned 0.76 W. The first pass pointed the same way, at +0.61 W. The extra CPU time cost little energy, since the CPU cores drew about the same either way (+4.01 and +4.14 W); the difference was in the iGPU and the rest of the chip. Nothing was reading Sepia Cam in these runs. In a call, Chrome encodes video on the CPU and keeps the iGPU awake to draw it, which could change the balance either way. `--post` stays `cpu` by default until both options are measured again with Meet reading Sepia Cam. Two problems remain on battery: the NPU path runs just under 30 fps, and latency from capture to send is about 100 ms against a 66 ms budget. Meet's own blur, the last baseline, hasn't been measured yet.

### Async inference and OpenCV threads (Oct 22, 2026)

`sepiad --bench 60 --infer NPU --post cpu` in 4 variants, with idle phases between them as in the matrix, 3 runs from 20:04 to 20:43 IST. On battery (39% at the start, 31% at the end), power profile `balance_power`, real camera, nothing reading Sepia Cam, OpenVINO 2025.1, `intel-npu-driver` 1.16.0. The synchronous variants are commit 2180c08. "1 thread" means `cv2.setNumThreads(1)` was called before sepiad started. The async variant was a trial, not committed: the NPU segmented frame N while the CPU blended and sent frame N−1. Idle package power was between 2.37 and 2.85 W. In run 2 the laptop suspended for about 10 minutes during the cooldown before an idle phase, and that phase read 2.85 W against about 2.45 W for the others. Leaving it out moves no result by more than 0.1 W; the table includes it.

Mean ± standard deviation of the 3 runs, power added over idle:

| Variant | Package | CPU cores | fps | Latency mean / p95 | sepiad CPU | Blur and blend | Waiting for the NPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sync, default threads (2180c08) | +6.02 ± 0.09 W | +3.77 W | 29.3 | 96.8 / 165.8 ms | 129% | 4.6 ms | 14.7 ms |
| sync, 1 thread | +5.00 ± 0.27 W | +2.88 W | 29.7 | 72.0 / 114.1 ms | 54% | 8.4 ms | 14.1 ms |
| async, default threads | +5.73 ± 0.12 W | +3.56 W | 29.8 | 88.8 / 117.9 ms | 135% | 4.7 ms | 0.8 ms |
| async, 1 thread | +5.39 ± 0.13 W | +3.14 W | 29.6 | 87.8 / 110.6 ms | 58% | 7.2 ms | 0.6 ms |

By default OpenCV splits each call across all 18 CPU threads. With one thread it runs each call on the calling thread, and blur and blend take 8.4 ms instead of 4.6 ms, which still fits in the frame. CPU use falls from 129% to 54% and package power by 1.02 W, nearly all of it in the CPU cores (+3.77 to +2.88 W). Mean latency falls from 96.8 to 72.0 ms.

Async inference cut the time spent waiting for the NPU from about 14 ms to under 1 ms, and with default threads it raised the frame rate from 29.3 to 29.8 fps. With one thread the synchronous loop already reaches 29.7 fps. There, async cost 0.39 W more and added 15.8 ms of mean latency, about the one frame it holds back by design.

**Decision:** sepiad calls `cv2.setNumThreads(1)` and keeps the synchronous loop. On battery the latency is still over budget: 72.0 ms mean and 114.1 ms at the 95th percentile, against 66 ms. The comparison of blur on the CPU and on the iGPU was made with default threads; with one thread the CPU side costs less, and the iGPU side hasn't been measured that way yet.

### Mask from the previous frame, plugged in (Oct 22, 2026)

Plugged in, with the CPU on the `balanced-battery` profile (`epp` is `balance_power` in every row). Real camera, nothing reading Sepia Cam, OpenVINO 2025.1, `intel-npu-driver` 1.16.0, `--infer NPU --post cpu`. The old loop is commit ab5ee2a, where each frame waits for its own mask. The new loop is commit d81e7b2: the NPU segments a frame while the CPU blends it with the previous frame's mask and sends it.

**Power A/B:** `bench_matrix.py` with both loops, 3 runs from 21:36 to 21:53 IST. Idle package power was 1.93 to 2.12 W, except the first idle phase (3.17 W), which overlapped the harness starting up and raises the old loop's baseline in run 1. Mean ± standard deviation, power added over idle:

| Loop | Package | CPU cores | fps | Latency mean / p95 | sepiad CPU | Decode | Blur and blend | Waiting for the NPU | Waiting for the camera |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old (ab5ee2a) | +4.74 ± 0.16 W | +2.51 W | 29.8 | 66.7 / 86.2 ms | 52% | 6.0 ms | 8.8 ms | 14.1 ms | 2.9 ms |
| new | +5.12 ± 0.23 W | +2.78 W | 29.7 | 51.1 / 60.9 ms | 54% | 7.8 ms | 7.6 ms | 0.6 ms | 16.2 ms |

The new loop cuts mean latency by 15.6 ms and the 95th percentile by 25.3 ms. Each frame now takes about 17.5 ms of processing, so sepiad waits about 16 ms for the next one and frames no longer queue. The other 33 ms or so pass before sepiad can read the frame: the driver timestamps the start of the frame, and the camera takes about a frame period to send it. The camera offers 720p MJPG only at 30 fps, so sepiad can't shorten that part. Decode slowed from 6.0 to 7.8 ms, probably because the CPU drops its clock during the longer wait.

The new loop costs 0.38 W more over idle. Without run 1, whose first idle phase read high, the gap is 0.22 W. Both are close to the spread of the runs. The async trial earlier the same day cost 0.39 W for a similar overlap, so the cost looks real but small.

**Check of the plugged-in method:** the old loop is the same code as "sync, 1 thread" in [Async inference and OpenCV threads](#async-inference-and-opencv-threads-oct-22-2026), which ran on battery. Package power agreed within the spread (+4.74 ± 0.16 W plugged in, +5.00 ± 0.27 W on battery), as did fps (29.8 and 29.7) and the NPU wait (14.1 ms in both). Plugged in, latency read lower (66.7 / 86.2 against 72.0 / 114.1 ms), decode was about 1 ms faster, and idle was about 0.4 W lower. Latency compared across the two methods can differ by a few ms; comparisons within one session are unaffected.

**Latency-only runs:** same conditions, 21:54 to 22:03 IST, `sepiad --bench 60` 3 times per variant, alternating, no idle phases. Package power here is the raw reading, not power over idle.

| Variant | fps | Latency mean / p95 | Package (raw) | Decode | Blur and blend |
| --- | --- | --- | --- | --- | --- |
| new loop | 29.8 | 50.7 / 56.5 ms | 7.33 W | 8.0 ms | 7.5 ms |
| new loop, 2 camera buffers | 29.5 | 51.9 / 65.7 ms | 7.22 W | 8.7 ms | 7.2 ms |
| new loop, `--post gpu` | 29.4 | 59.4 / 136.7 ms | 8.49 W | 11.6 ms | 5.8 ms |

With frames no longer queueing, two camera buffers instead of OpenCV's default four change nothing; the higher p95 comes from one run with a stall (87.6 ms). On the iGPU, blur and blend take 1.7 ms less, but decode takes about 3.6 ms more, the tail is much longer, and the package draws 1.16 W more. With one OpenCV thread, the CPU path is ahead on every number here.

**On power saver:** earlier that evening, at 20% battery, GNOME had switched to power saver (`epp` is `power`). In two 60 s runs each, the old loop fell to 21.5 fps with 215.8 ms mean and 235.1 ms p95 latency, while the new loop held 29.8 fps with 63.0 ms mean and 76.2 ms p95.

**Decision:** sepiad blends each frame with the previous frame's mask. The camera keeps OpenCV's default buffers, and blur and blend stay on the CPU. Latency is 51.1 ms mean and 60.9 ms p95, of which about 33 ms is the camera's own frame time.

### Blur modes and edge levels, plugged in (Oct 22 and 23, 2026)

Plugged in, with the CPU on the `balanced-battery` profile (`epp` is `balance_power` in every row). Real camera, nothing reading Sepia Cam, OpenVINO 2025.1, `intel-npu-driver` 1.16.0, `--infer NPU --post cpu`, Standard blur unless named. The latency-only runs are `sepiad --bench 60` 3 times per variant, alternating, and their package power is the raw reading. "Mask and edge" is the `mask_ms` column: temporal smoothing plus the edge level. The bench.csv header committed in d81e7b2 had a stray carriage return before `epp`; it is fixed and also names the new `blur` and `edges` columns.

**First versions,** Oct 22, 23:44 to 23:59 IST. High did its last steps at 720p, and Portrait blurred a 640×360 copy with a 9×9 kernel. "Buffers kept" set glibc's `M_MMAP_THRESHOLD` and `M_TRIM_THRESHOLD` to 1 GiB, so freed frame-sized buffers stay in the heap.

| Variant | fps | Latency mean / p95 | Package (raw) | Mask and edge | Blur and blend |
| --- | --- | --- | --- | --- | --- |
| medium, buffers not kept | 29.8 | 51.5 / 65.4 ms | 8.00 W | 0.23 ms | 8.03 ms |
| medium, buffers kept | 29.8 | 52.0 / 68.2 ms | 7.97 W | 0.23 ms | 8.06 ms |
| low, buffers kept | 29.8 | 51.6 / 66.4 ms | 8.07 W | 0.16 ms | 8.09 ms |
| high at 720p, buffers kept | 29.3 | 60.4 / 100.2 ms | 8.75 W | 4.91 ms | 6.54 ms |
| Portrait at 640×360, medium, buffers kept | 29.8 | 53.7 / 77.4 ms | 8.01 W | 0.24 ms | 9.08 ms |

In an offline timing on a synthetic frame, keeping buffers took blur and blend from 9.3 to 5.4 ms and high from 7.9 to 3.2 ms. With the camera running it changed nothing, so sepiad doesn't set it. Low and medium cost the same.

**Power A/B,** Oct 22, 23:59 to Oct 23, 00:16 IST: `bench_matrix.py`, 3 runs, commit 5070788 (low edges) against medium edges with buffers kept. Idle package power was 2.96 to 3.43 W, about 1 W higher than earlier that evening and less steady. Mean ± standard deviation, power added over idle:

| Variant | Package | CPU cores | fps | Latency mean / p95 |
| --- | --- | --- | --- | --- |
| 5070788, low | +4.72 ± 0.11 W | +2.85 W | 29.6 | 54.0 / 75.4 ms |
| medium, buffers kept | +5.05 ± 0.15 W | +3.06 W | 29.8 | 51.4 / 64.4 ms |

The 0.33 W gap is about twice the spread. The latency-only runs, which alternated the variants within minutes, show no power difference between low and medium or with buffers kept, so the unsteady idle is the more likely cause.

**Cheaper high and Portrait,** Oct 23, 10:43 to 10:52 IST, buffers not kept. High does its full-size steps at 640×360, and Portrait blurs the same 320×180 copy as Standard with a 5×5 kernel. Package power leaves out the first round, where the default's run read 10.17 W.

| Variant | fps | Latency mean / p95 | Package (raw) | Mask and edge | Blur and blend |
| --- | --- | --- | --- | --- | --- |
| medium, Standard (default) | 29.8 | 51.9 / 63.8 ms | 7.81 W | 0.21 ms | 7.38 ms |
| high, Standard | 29.5 | 55.7 / 82.3 ms | 7.83 W | 2.58 ms | 6.72 ms |
| medium, Portrait | 29.8 | 51.5 / 67.9 ms | 7.70 W | 0.21 ms | 7.05 ms |

The p95 moved by about 20 ms between rounds for every variant, and the first round was the highest for all three, so Portrait and the default are equal within the spread. High's p95 was over 66 ms in every round (96.9, 72.5 and 77.4 ms).

**Decision:** medium edges and Standard blur are the defaults; medium was picked in a live preview and costs the same as low. Portrait costs the same as Standard. High stays available at about 4 ms more mean latency, with its p95 over the 66 ms target. None of the levels changed package power measurably. The cheaper versions are commit ea017e1.

### Meet comparison and Meet-style blur, plugged in (Oct 23, 2026)

Plugged in, with the CPU on the `balanced-battery` profile (`epp` is `balance_power` in every row). OpenVINO 2025.1, `intel-npu-driver` 1.16.0, sepiad with `--infer NPU`, medium edges.

**Meet comparison,** Oct 23, 11:42 to 12:03 IST: `scripts/bench_meet.py`, 3 runs, in a Meet call in Chrome with the microphone on in every phase. Each phase measured the whole laptop for 60 s, after 20 s for Meet to settle. "Meet's blur" is Meet's full background blur on the real camera. In the Sepia Cam rows Meet reads Sepia Cam with its own effects off, and fps and latency come from sepiad over the same window. Laptop CPU counts every process (100% is one core). These runs used the old Standard blur, 15×15 over the whole frame. Mean ± standard deviation:

| Phase | Package | Over Meet, no effects | CPU cores | iGPU | Laptop CPU | fps | Latency mean / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Meet, no effects | 8.93 ± 0.05 W | | 5.81 W | 0.16 W | 131% | | |
| Meet's blur | 11.29 ± 0.09 W | +2.36 ± 0.06 W | 6.35 W | 1.03 W | 180% | | |
| Sepia Cam, `--post cpu` | 12.28 ± 0.17 W | +3.35 ± 0.13 W | 7.37 W | 0.16 W | 169% | 29.7 | 54.0 / 84.2 ms |
| Sepia Cam, `--post gpu` | 12.47 ± 0.15 W | +3.54 ± 0.12 W | 6.95 W | 0.48 W | 159% | 29.7 | 52.5 / 77.7 ms |

Meet's blur costs about 1 W less than Sepia Cam. Over Meet with no effects, Meet's blur adds 0.54 W on the CPU cores, 0.87 W on the iGPU and 0.95 W in the rest of the package. Sepia Cam with `--post cpu` adds 1.56 W on the cores, nothing on the iGPU and 1.79 W in the rest. In a check after the runs, Chrome had only the iGPU's render node open, not the NPU, and the NPU's busy counter didn't move. That fits Google's description of Meet's HD segmentation running on the GPU through WebGL ([Google Research](https://research.google/blog/high-definition-segmentation-in-google-meet/)). Where Sepia's extra watt goes is not measured yet: sepiad decodes the camera's MJPEG on the CPU, and Chrome converts the BGR frames it reads from Sepia Cam. A Meet phase with `sepiad --infer none` would show how much of the gap that pipeline accounts for.

`--post cpu` used 0.19 W less than `--post gpu`, and `--post gpu` had a 6.5 ms lower p95. Both gaps are about one standard deviation, so where blur and blend run is still open. With either option the p95 went over 66 ms during the call, while the mean stayed at 52 to 54 ms.

**Meet-style blur,** Oct 23, 12:29 to 12:35 IST. Real camera, nothing reading Sepia Cam, `sepiad --bench 60` 3 times per variant, alternating. Three changes, after Google's description of Meet's blur ([Google Research](https://research.google/blog/background-features-in-google-meet-powered-by-web-ml/)) and a look at Meet's own preview:

- The blur weights each pixel by 1 − mask, so only the background goes into it and the person's colours no longer bleed into a halo around them.
- Standard is 21×21 instead of 15×15. It was picked in a live preview, after 31×31 looked too strong.
- High's guided filter averages over a 5×5 window at 320×180 instead of 9×9, with eps 20 instead of 65. In a live preview it followed hair more closely, and in an offline timing it cost the same 3.1 ms.

Offline, on a synthetic frame with one OpenCV thread, the new blur and blend took 12.30 ms and the old one 11.18 ms.

| Variant | fps | Latency mean / p95 | Package (raw) | Mask and edge | Blur and blend |
| --- | --- | --- | --- | --- | --- |
| medium, Standard (default) | 29.8 | 51.1 / 56.2 ms | 7.54 W | 0.21 ms | 8.06 ms |
| high, Standard | 29.4 | 56.6 / 86.5 ms | 7.32 W | 2.72 ms | 7.33 ms |

The default's p95 stayed under 59 ms in every run, the lowest so far, although blur and blend take about 0.7 ms longer than in the previous entry. High's p95 was 55.5, 125.3 and 78.6 ms, and the 125.3 ms run dropped to 28.9 fps, so high still goes over the target.

**Decision:** the background-only blur and the 21×21 Standard become the default. Medium edges stay the default and high stays opt-in. The Meet script and the blur changes are commit 3f777e2.

### Sepia Cam as a passthrough in a Meet call, plugged in (Oct 23, 2026)

Oct 23, 13:11 to 13:30 IST: `scripts/bench_meet.py 3 60 meet meet+NPU:cpu meet+none`, 3 runs, in a Meet call in Chrome, as in the [Meet comparison](#meet-comparison-and-meet-style-blur-plugged-in-oct-23-2026). Plugged in, with the CPU on the `balanced-battery` profile (`epp` is `balance_power` in every row). OpenVINO 2025.1, `intel-npu-driver` 1.16.0, code at commit 9fa1f90: Standard blur, medium edges. In the passthrough phase sepiad runs with `--infer none`, so it decodes the camera's MJPEG and sends each frame to Sepia Cam unchanged, with no model and no blur. "Rest" is package minus CPU cores minus iGPU, where the NPU's power is counted. Mean ± standard deviation:

| Phase | Package | Over Meet, no effects | CPU cores | iGPU | Rest | Laptop CPU | fps | Latency mean / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meet, no effects | 8.54 ± 0.20 W | | 5.48 W | 0.17 W | 2.90 W | 105% | | |
| Sepia Cam, passthrough | 8.84 ± 0.20 W | +0.30 ± 0.01 W | 5.66 W | 0.17 W | 3.02 W | 131% | 29.4 | 55.8 / 148.8 ms |
| Sepia Cam, `--post cpu` | 11.79 ± 0.37 W | +3.25 ± 0.18 W | 6.83 W | 0.17 W | 4.79 W | 144% | 29.7 | 52.5 / 73.5 ms |

The pipeline costs little. Capture, CPU decode, sending BGR frames to Sepia Cam and Chrome reading them add 0.30 W to the call, although sepiad uses about a quarter of a core for it. Sending I420 instead of BGR or decoding on the iGPU could save at most that 0.30 W.

The model and the blur add 2.95 W on top of the passthrough: 1.17 W on the CPU cores and 1.77 W in the rest of the package. Inference on the NPU alone took 1.95 W at 30 inferences per second in the [model-only benchmark](#model-only-power-oct-22-2026), close to the 2.36 W that Meet's whole blur added in the Meet comparison (a different session). The 16 MB model is the main cost. Meet's segmenter for its earlier blur has 193K parameters, about 400 KB ([Google Research](https://research.google/blog/background-features-in-google-meet-powered-by-web-ml/)).

The passthrough's p95 was 118.7, 186.7 and 141.1 ms. With less work, the CPU ran at lower clocks on `balance_power`: decode took 13.1 to 13.3 ms against 8.3 to 8.7 ms with the model running. With the background-only blur, the in-call p95 was 72.6 to 74.8 ms, under the 78 to 84 ms of the old blur but still over 66 ms.

**Decision:** Sepia Cam's extra watt in a call comes from the model and the blur, so I420 output and iGPU decode are dropped as power levers. The levers left are running the model on fewer frames and getting the 250 KB selfie model to load (see [ROADMAP.md](ROADMAP.md#open-questions)).

### Smaller models and crisp edges, plugged in (Oct 23, 2026)

Plugged in, with the CPU on the `balanced-battery` profile (`epp` is `balance_power` in every row). OpenVINO 2025.1, `intel-npu-driver` 1.16.0.

**Model-only timing,** offline, 30 inferences per device after one warm-up. The multiclass model and meet_1655 ran on a random 1280×720 frame through `scripts/preview_models.py`, so their times include the resize, and for the multiclass model a softmax in NumPy; they are higher than sepiad's built-in path. PP-HumanSeg ran on a random −1 to 1 input with no resize. meet_1655 and ML Kit's 256×256 selfie model load once their custom operation is rewritten ([ARCHITECTURE.md](ARCHITECTURE.md#the-multiclass-model-with-pp-humanseg-as-the-fallback)). The ML Kit model has the same 106,401 weights as meet_1655 and wasn't timed.

| Model | Input | File | Weights | NPU | iGPU | CPU |
| --- | --- | --- | --- | --- | --- | --- |
| MediaPipe Selfie Multiclass | 256×256 | 16 MB | | 26.28 ms | 15.40 ms | 67.05 ms |
| meet_1655 (MediaPipe landscape) | 256×144 | 250 KB | 106,401 | 1.88 ms | 1.58 ms | 3.03 ms |
| PP-HumanSeg v2 Lite, portrait | 256×144 | 1.9 MB (OpenVINO IR) | 937,025 | 4.52 ms | 1.29 ms | 7.73 ms |

**Live previews,** judged by the user. Next to the multiclass model in `scripts/preview_models.py`, meet_1655 outlined the face well in good light but lost the hand, and ML Kit's model did better than meet_1655 in low light. PP-HumanSeg held the hand and stayed steady in both good and low light, with a looser outline. The multiclass model beat all three on outline and stability. With sepiad's blur, PP-HumanSeg's result looked much closer to Meet's, but its edge around the hand was soft. With the fingers spread, it blurred some fingers or filled in the gaps between them, and high edges didn't change that.

A new edge level, `crisp`, enlarges the mask to 640×360 before steepening its edge four times around 0.5, so the outline stays sharp at full size. On PP-HumanSeg it looked like a fixed, rough outline following the person around, though it handled the hand a little better. On the multiclass model it showed no jitter and looked better than medium, in some ways better than Meet's own blur.

**Latency-only runs,** Oct 23, 15:00 to 15:10 IST: real camera, nothing reading Sepia Cam, `sepiad --bench 60` 3 times per variant, alternating, Standard blur, inference on the NPU, no idle phases. Package power is raw. sepiad CPU is sepiad's own (100% is one core).

| Variant | fps | Latency mean | p95 per run | Package | CPU cores | Mask and edge | Blur and blend | NPU busy | sepiad CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiclass, crisp | 29.7 | 52.0 ms | 54.7, 89.0, 54.1 ms | 7.16 ± 0.10 W | 3.29 W | 0.60 ms | 7.73 ms | 37% | 55% |
| multiclass, medium | 29.3 | 55.6 ms | 88.0, 92.2, 54.1 ms | 7.11 ± 0.07 W | 3.24 W | 0.22 ms | 8.11 ms | 38% | 56% |
| PP-HumanSeg, crisp | 29.6 | 56.6 ms | 76.9, 78.7, 87.3 ms | 6.28 ± 0.03 W | 3.72 W | 0.53 ms | 10.45 ms | 20% | 63% |

Crisp costs the same as medium. Package power is within the spread and the mask step takes 0.4 ms more. Both multiclass variants had runs with a p95 near 90 ms and runs at 54 ms, so the p95 doesn't separate them.

PP-HumanSeg used 0.88 W less package power than the multiclass model with crisp edges, and kept the NPU busy 20% of the time against 37%. Its CPU cores drew 0.43 W more, and blur and blend took 10.45 ms against 7.73 ms in every run. Offline, with inference running alongside the blur as in the loop, the blur took about the same time with either model (10.84 and 11.14 ms), and inference used 0.13 ms of CPU per frame, so the cause isn't known yet. Its p95 was over 66 ms in all three runs.

**Decision:** crisp becomes the default edge level, and the multiclass model stays the default model. PP-HumanSeg v2 Lite is the fallback, `--model humanseg`: it fits in a frame on the CPU, and it is the lower-power choice, although its saving in a call isn't measured yet. meet_1655 and the ML Kit model aren't used. These rows have the new `model` column. The changes are commit b30639c.

### PP-HumanSeg against the multiclass model in a Meet call, plugged in (Oct 24, 2026)

Oct 24, 15:06 to 15:28 IST: `scripts/bench_meet.py 3 60 meet-blur meet meet+NPU:cpu meet+humanseg:cpu`, 3 runs, in a Meet call in Chrome with the microphone on in every phase and no other participant. Plugged in, with the CPU on the `balanced-battery` profile (`epp` is `balance_power` in every row). OpenVINO 2025.1, `intel-npu-driver` 1.16.0, sepiad at commit 9f31400: Standard blur, crisp edges, inference on the NPU, blur and blend on the CPU. The PP-HumanSeg phase runs sepiad with `--model humanseg`. "Rest" is package minus CPU cores minus iGPU, where the NPU's power is counted. Mean ± standard deviation:

| Phase | Package | Over Meet, no effects | CPU cores | iGPU | Rest | Laptop CPU | fps | Latency mean / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meet, no effects | 8.29 ± 0.04 W | | 5.26 W | 0.17 W | 2.87 W | 110% | | |
| Meet's blur | 10.54 ± 0.02 W | +2.25 ± 0.06 W | 5.65 W | 1.04 W | 3.86 W | 149% | | |
| Sepia Cam, multiclass | 11.49 ± 0.08 W | +3.20 ± 0.07 W | 6.56 W | 0.17 W | 4.76 W | 154% | 29.8 | 52.5 / 69.9 ms |
| Sepia Cam, PP-HumanSeg | 10.38 ± 0.07 W | +2.09 ± 0.10 W | 6.80 W | 0.18 W | 3.40 W | 170% | 29.7 | 56.4 / 77.5 ms |

Run by run, PP-HumanSeg used 1.11 ± 0.07 W less package power than the multiclass model and 0.16 ± 0.08 W less than Meet's own blur, so in a call it costs about what Meet's blur does. The whole saving is in the rest of the package, where the NPU sits, which drew 1.36 W less. The CPU cores drew 0.24 W more, laptop CPU went from 154% to 170%, and the NPU was busy 20% of the time against 38%.

As in the latency-only runs, blur and blend were slower with PP-HumanSeg, 11.23 ms against 8.10 ms in every run, and the cause still isn't known. Its p95 was 77.5, 76.1 and 78.9 ms. The multiclass model's was 61.7, 74.5 and 73.4 ms, over 66 ms in two runs of three.

Meet's blur looked steadier than Sepia Cam's to the user. Once it settled, its outline stayed the same until they moved, and then it followed the movement well.

The [Meet comparison](#meet-comparison-and-meet-style-blur-plugged-in-oct-23-2026) of Oct 23 had a second participant in the call, a phone; the [passthrough session](#sepia-cam-as-a-passthrough-in-a-meet-call-plugged-in-oct-23-2026) and this one had none. Meet with no effects drew 8.93, 8.54 and 8.29 W in the three sessions. So the earlier comparison of Meet's blur (+2.36 W) with Sepia Cam's background-only blur (+3.25 W) mixed two setups; this entry measures all of them in one.

**Decision:** the multiclass model stays the default for its outline, and `--model humanseg` is the low-power option, at about the cost of Meet's blur. Next is a steadier mask that holds still while the person does, as Meet's does.

### Steady mask smoothing, plugged in (Oct 24, 2026)

Oct 24, 15:47 to 15:53 IST: real camera, nothing reading Sepia Cam, `sepiad --bench 60` 3 times per variant, alternating. Multiclass model on the NPU, Standard blur, crisp edges. Plugged in, with the CPU on the `balanced-battery` profile (`epp` is `balance_power` in every row). OpenVINO 2025.1, `intel-npu-driver` 1.16.0. Package power is raw, and sepiad CPU is sepiad's own (100% is one core).

`--smooth steady` replaces the fixed average of 60% new mask and 40% previous. Each pixel takes in a share of the new mask in proportion to how much it changed: 10% for a change up to 0.03, which is model noise while the person sits still, and all of it from 0.3 up, when they move. It follows what the user saw of Meet's blur in the [Meet call](#pp-humanseg-against-the-multiclass-model-in-a-meet-call-plugged-in-oct-24-2026): an outline that holds still until the person moves. In a live preview with both models, the user found steady very good.

| Smoothing | fps | Latency mean | p95 per run | Package | CPU cores | Infer stage | Blur and blend | sepiad CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 29.8 | 51.3 ms | 55.7, 59.0, 55.1 ms | 7.94 ± 0.18 W | 3.83 W | 0.54 ms | 7.37 ms | 55% |
| steady | 29.8 | 51.4 ms | 56.0, 56.7, 55.5 ms | 7.83 ± 0.05 W | 3.75 W | 0.67 ms | 7.38 ms | 56% |

The smoothing is timed in the infer stage, after the wait for the mask, and steady adds 0.12 ms there. Latency and package power are the same within the spread; fixed's higher mean comes from its first run, at 8.15 W. Every p95 stayed under 60 ms.

**Decision:** steady becomes the default smoothing, and `--smooth fixed` stays available. It hasn't been measured in a call. The change is commit 7fa48b6.

### Where the latency goes (Oct 24, 2026)

No new runs. Each of the 12 sepiad rows of Oct 24, from the [Meet call](#pp-humanseg-against-the-multiclass-model-in-a-meet-call-plugged-in-oct-24-2026) and the [steady smoothing runs](#steady-mask-smoothing-plugged-in-oct-24-2026), splits its latency into two parts. The first is the time from the driver's timestamp until sepiad dequeues the frame, which is the latency minus the stage times after the read. The second is sepiad's own processing, decode to send. Ranges of the per-run means:

| Rows | Latency | Timestamp to dequeue | Decode to send | p95 minus 33.3 ms |
| --- | --- | --- | --- | --- |
| Camera only, multiclass (6 runs) | 51.1 to 51.6 ms | 33.2 to 33.5 ms | 17.7 to 18.2 ms | 21.8 to 25.7 ms |
| Meet call, multiclass (3 runs) | 51.8 to 53.3 ms | 33.4 to 34.1 ms | 18.4 to 19.2 ms | 28.4 to 41.2 ms |
| Meet call, PP-HumanSeg (3 runs) | 56.0 to 56.9 ms | 34.0 to 34.4 ms | 22.0 to 22.5 ms | 42.8 to 45.6 ms |

In every run the frame reached sepiad about one frame period after its driver timestamp, so on average no frame waited in the queue. That time is the camera sending the frame, and Chrome waits just as long when it reads the real camera. Sepia's own share is the rest: 17.7 to 19.2 ms on average with the multiclass model and 22.0 to 22.5 ms with PP-HumanSeg. The last column subtracts one frame period from the p95, which assumes the camera's part stays at its mean; its run means varied by 1.2 ms.
