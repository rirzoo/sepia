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

## Metrics and how they're measured on this laptop

| Metric | Source | Root? |
| --- | --- | --- |
| Chip power by domain | RAPL energy counters in `/sys/class/powercap`: package, CPU cores, iGPU (`uncore`) and platform (`psys`), read at the start and end of each phase. The NPU has no domain of its own; its power is in package minus cores minus iGPU | Yes, or open them until reboot ([SETUP.md](SETUP.md), step 6) |
| Whole-system power | `/sys/class/power_supply/BAT1/power_now`, sampled at 1 Hz, on battery. Cross-check only: idle readings swung between 5.4 and 11.0 W, too noisy for differences of a few watts | No |
| NPU utilisation | Change in `/sys/class/accel/accel0/device/npu_busy_time_us` over the run | No |
| Time per stage | `time.perf_counter()` around each step | No |
| fps and latency (mean, p95) | Capture and send timestamps | No |
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
