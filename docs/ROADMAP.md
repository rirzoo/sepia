# Roadmap

Where Sepia is going, what could go wrong, what's still unknown, and what's next.

The project grows in four levels; the focus is Levels 1 and 2, which are small enough to finish and produce a polished tool plus real Linux NPU/iGPU/CPU benchmark data.

## Level 1: Personal tool plus benchmark (focus)

**Scope:** About 500–700 lines of Python, including sepiad and sepiactl. Works on this laptop.

1. Setup checks pass ([SETUP.md](SETUP.md), prerequisites 1–4).
2. Blur-only pipeline with `--infer`, `--post` and `--bench`. This gives both the core of v1 and the benchmark harness.
3. Run the benchmark matrix and baselines ([BENCHMARK.md](BENCHMARK.md)); choose default devices from the data.
4. `sepiactl` over a Unix socket; `sepiad` as a systemd user service.
5. Automatic framing from the mask's person box, with a smoothed crop window.
6. Portrait light on the mask's face-skin region.

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

The biggest open question is whether the NPU uses less power than the iGPU for this workload. The benchmark answers it in Level 1 step 3, before most of the code is written.

- **The NPU may not win.** It is slightly slower than the iGPU per frame (8.7 vs 7.2 ms), so it wins only if it uses less power. If the iGPU measures better, it becomes the default and the benchmark reports it. Either result is useful.
- **16 MB model, no CPU fallback.** The multiclass model is about 60× the size of the small selfie models and takes 52 ms on the CPU, so a laptop without a working NPU or iGPU can't run it at 30 fps. Mitigation: rewrite the small models' custom operation (see Open questions).
- **CPU-bound blur and blend.** MJPEG decode, blur and blend may cost more than inference. Mitigations: downscaled blur, `blendLinear`, iGPU via OpenCL.
- **Pinned old NPU driver.** Sepia needs `intel-npu-driver` 1.16.0 to match Fedora's OpenVINO 2025.1. Intel's published test coverage for Meteor Lake (about 83% of the extended suite) is for 1.38.0 ([notes](https://www.linuxcompatible.org/story/intel-npu-driver-1380-adds-ubuntu-2604-lts-support-and-major-kernel-70-upgrade/)). The current model runs on 1.16; later models may not. Removing the version lock before Fedora updates OpenVINO breaks the NPU again. Other users of Level 2 packages will hit the same mismatch.
- **Darker image at a fixed 30 fps.** With dynamic frame rate off, exposure is capped at 33 ms, so the picture is darker in low light (mean brightness 82/255 in an evening room). Portrait light and the brightness-normalisation step compensate. The mask also holds up much better in a dark room with −1 to 1 input scaling than with 0 to 1.
- **Mask lag vs flicker.** Temporal smoothing stops flicker but makes edges trail fast movement; the 0.6/0.4 blend needs tuning.
- **v4l2loopback on new kernels.** The akmod sometimes lags new kernel releases (this laptop runs 7.2); a kernel update can temporarily remove the virtual camera.
- **Noisy power readings.** Battery `power_now` measures the whole system. Fixed conditions, repeated runs and baseline subtraction keep the comparison honest.

## Open questions

- Is the multiclass model fast and cheap enough end to end, or is it worth rewriting the small models' `Convolution2DTransposeBias` as a standard transposed convolution plus bias, to get the 250 KB model back (and a viable CPU path)?
- Which on-demand detection works with Chrome: the v4l2loopback event or `/proc` polling?

## Next steps

- [x] Enable RPM Fusion free and install the prerequisites (setup steps 2–4)
- [x] Confirm OpenVINO lists `CPU`, `GPU`, `NPU`, OpenCV has OpenCL, and the model runs on all three
- [x] Load v4l2loopback, confirm `/dev/video10`, and pin the NPU driver
- [x] End-to-end check: real 720p MJPG frames → model on the NPU → `/dev/video10` → read back by a second process (pyvirtualcam 0.15.0 installs cleanly on Python 3.14)
- [x] Settle the input scaling: −1 to 1 ([ARCHITECTURE.md](ARCHITECTURE.md#per-frame-loop-30-times-a-second), step 2)
- [ ] Write the blur-only sepiad with `--infer`, `--post` and `--bench`
- [ ] Run the baselines and the 6-configuration matrix; choose default devices
- [ ] Add sepiactl and the systemd user service
- [ ] Add automatic framing, then portrait light
