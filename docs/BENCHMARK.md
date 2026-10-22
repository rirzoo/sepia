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

On battery, with screen brightness and Wi-Fi left as they are and no new work started during the run. Steady background apps can stay open. Each configuration runs for 60 s, and idle phases alternate with the configurations; each configuration is compared with the mean of the idle phases on either side of it, so slow changes in background load cancel out. Each idle phase starts after a 15 s cooldown, because power stays high for several seconds after a load ends. The spread of the idle phases is the noise floor, and smaller differences don't count. The whole sequence runs 3 times, and results are reported as mean ± standard deviation. Raw numbers and the method are published with the README.

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
