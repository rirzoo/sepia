# Roadmap

Where Sepia is going, what could go wrong, what's still unknown, and what's next.

The project grows in four levels; the focus is Levels 1 and 2, which are small enough to finish and produce a polished tool plus real Linux NPU/iGPU/CPU benchmark data.

## Level 1: Personal tool plus benchmark (focus)

**Scope:** About 500–700 lines of Python, including sepiad and sepiactl. Works on this laptop.

1. Setup checks pass ([SETUP.md](SETUP.md), prerequisites 1–4).
2. Blur-only pipeline with `--infer`, `--post` and `--bench`. This gives both the core of v1 and the benchmark harness.
3. Run the benchmark matrix and baselines ([BENCHMARK.md](BENCHMARK.md)); choose default devices from the data.
4. Blur modes and edge levels ([OVERVIEW.md](OVERVIEW.md#feature-set)): Standard and a lighter Portrait blur, and edges at low, medium or high. Low is today's mask scaled up. Medium sharpens the mask's edge at 256×256, which costs almost nothing. High refines the edge against the full frame with a guided filter, on the CPU or the iGPU. Each level is measured in ms and W, and each has to stay inside the latency target. Open question: should high use the frame's own mask instead of the previous frame's?
5. `sepiactl` over a Unix socket; `sepiad` as a systemd user service.
6. Automatic framing from the mask's person box, with a smoothed crop window.
7. Portrait light on the mask's face-skin region.

**Done when:** Meet and WhatsApp Web show "Sepia Cam" at a steady 30 fps with at most 66 ms added latency; the benchmark table is recorded; and sepiad's power with Meet's blur off is compared against Meet's own blur, whichever way the result goes.

## Level 2: A tool other people can use (focus)

**Scope:** Builds on Level 1 so others can install and use it.

- **On-demand camera:** placeholder frames while idle, real camera only while an app reads ([ARCHITECTURE.md](ARCHITECTURE.md#on-demand-camera-after-v1))
- **Settings UI:** a GNOME/KDE tray app with effect toggles and a live preview, talking to sepiad the same way sepiactl does
- **Hardware detection:** pick the camera node automatically; use the measured default devices, falling back to whatever device exists; support USB and IPU6/MIPI cameras
- **Packaging:** a COPR/RPM first, then Flatpak and AUR
- **Benchmarks:** `sepiad --bench` runnable by anyone, so results from other laptops can be collected
- **Tests and docs:** install guide, troubleshooting for the camera and NPU driver

**Done when:** someone else with a Core Ultra laptop installs it from a package and it works without editing code.

## Level 3: A proper Linux project (if people adopt it)

Replace v4l2loopback with a native PipeWire virtual camera (better with Flatpak apps and sandboxed browsers), offer the effects as a GStreamer element, support other NPUs through ONNX Runtime backends (for example AMD Ryzen AI), and bundle voice focus as a PipeWire filter.

## Level 4: Research-level features (separate project)

Real-time eye-contact correction on an 11 TOPS NPU, model-based relighting, and INT8 quantisation studies of quality vs speed vs power. This suits a technical report or undergrad research project.

## Risks

With the full blur pipeline on battery, NPU inference adds 6.19 W of package power against 10.12 W for the iGPU ([BENCHMARK.md](BENCHMARK.md#blur-pipeline-power-matrix-oct-22-2026)). The biggest open question now is whether that beats Meet's own blur.

- **Latency on battery.** With the `balance_power` profile the CPU stays at low clocks; with no effects the camera path even falls to about 26 fps with 129 ms of latency, because MJPEG decode takes 24 to 31 ms. Blending each frame with the previous frame's mask brought the NPU pipeline to 29.7 fps with 51.1 ms of latency on average and 60.9 ms at the 95th percentile ([BENCHMARK.md](BENCHMARK.md#mask-from-the-previous-frame-plugged-in-oct-22-2026)). About 33 ms of that is the camera's own frame time. Fewer camera buffers and blur on the iGPU didn't help. Options left: a cheaper blend, or blending only near the mask's edge.
- **16 MB model, no CPU fallback.** The multiclass model is about 60× the size of the small selfie models and takes 52 ms on the CPU, so a laptop without a working NPU or iGPU can't run it at 30 fps. Mitigation: rewrite the small models' custom operation (see Open questions).
- **Pinned old NPU driver.** Sepia needs `intel-npu-driver` 1.16.0 to match Fedora's OpenVINO 2025.1. Intel's published test coverage for Meteor Lake (about 83% of the extended suite) is for 1.38.0 ([notes](https://www.linuxcompatible.org/story/intel-npu-driver-1380-adds-ubuntu-2604-lts-support-and-major-kernel-70-upgrade/)). The current model runs on 1.16; later models may not. Removing the version lock before Fedora updates OpenVINO breaks the NPU again. Other users of Level 2 packages will hit the same mismatch.
- **Darker image at a fixed 30 fps.** With dynamic frame rate off, exposure is capped at 33 ms, so the picture is darker in low light (mean brightness 82/255 in an evening room). Portrait light and the brightness-normalisation step compensate. The mask also holds up much better in a dark room with −1 to 1 input scaling than with 0 to 1.
- **Mask lag vs flicker.** Temporal smoothing stops flicker but makes edges trail fast movement; the 0.6/0.4 blend needs tuning.
- **v4l2loopback on new kernels.** The akmod sometimes lags new kernel releases (this laptop runs 7.2); a kernel update can temporarily remove the virtual camera.
- **Noisy power readings.** Battery `power_now` swung between 5.4 and 11.0 W at idle, so measurements use the chip's RAPL counters instead, with alternating idle phases to cancel background load. The counters are root-only and have to be opened again after every reboot ([SETUP.md](SETUP.md), step 6).

## Open questions

- Is the multiclass model fast and cheap enough end to end, or is it worth rewriting the small models' `Convolution2DTransposeBias` as a standard transposed convolution plus bias, to get the 250 KB model back (and a viable CPU path)?
- Which on-demand detection works with Chrome: the v4l2loopback event or `/proc` polling?

## Next steps

- [x] Enable RPM Fusion free and install the prerequisites (setup steps 2–4)
- [x] Confirm OpenVINO lists `CPU`, `GPU`, `NPU`, OpenCV has OpenCL, and the model runs on all three
- [x] Load v4l2loopback, confirm `/dev/video10`, and pin the NPU driver
- [x] End-to-end check: real 720p MJPG frames → model on the NPU → `/dev/video10` → read back by a second process (pyvirtualcam 0.15.0 installs cleanly on Python 3.14)
- [x] Settle the input scaling: −1 to 1 ([ARCHITECTURE.md](ARCHITECTURE.md#per-frame-loop-30-times-a-second), step 2)
- [x] Model-only power benchmark, 3 runs: inference runs on the NPU ([ARCHITECTURE.md](ARCHITECTURE.md#inference-on-the-npu))
- [x] Write the blur-only sepiad with `--infer`, `--post` and `--bench`: 29.8 fps with inference on the NPU, about 22 ms of added latency
- [x] Run the camera-only baseline and the 6-configuration matrix on battery; inference stays on the NPU; blur and blend run on the CPU for now, a near tie with the iGPU ([ARCHITECTURE.md](ARCHITECTURE.md#blur-and-blend-on-the-cpu-provisionally))
- [x] Try async inference and one OpenCV thread on battery: one thread saves 1.02 W and keeps 29.7 fps; async is dropped
- [x] Measure plugged in with the CPU on the battery profile; it matches battery runs within their spread ([BENCHMARK.md](BENCHMARK.md#method))
- [x] Blend each frame with the previous frame's mask: 51.1 ms mean and 60.9 ms p95 latency, from 66.7 and 86.2 ms; fewer camera buffers and iGPU blur didn't help
- [ ] Decide whether a 60.9 ms p95 is far enough inside 66 ms, or cut blend time further (see Risks)
- [ ] Add the blur modes and edge levels (Level 1, item 4), on their own branch
- [ ] Measure Meet's own blur against Meet using Sepia Cam, running Sepia Cam with both `--post cpu` and `--post gpu`, and settle where blur and blend run
- [ ] Add sepiactl and the systemd user service
- [ ] Add automatic framing, then portrait light
