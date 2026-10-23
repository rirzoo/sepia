# Architecture

How sepiad and sepiactl work, and the decisions behind the design.

## How sepiad works

sepiad runs a one-time setup, then a loop that runs on every frame. Inference runs on the device chosen by `--infer`; blur and blend run on the CPU or the iGPU, chosen by `--post`. By default inference runs on the NPU, chosen by measurement ([Inference on the NPU](#inference-on-the-npu)), and blur and blend run on the CPU for now ([Blur and blend on the CPU, provisionally](#blur-and-blend-on-the-cpu-provisionally)).

```
capture (MJPG) → preprocess → infer [NPU | GPU | CPU] → mask post-process → effects [CPU | iGPU] → virtual camera
```

### Setup (once at start)

1. **Load the model.** MediaPipe Selfie Multiclass (`selfie_multiclass_256x256.tflite`, 16 MB, 256×256 RGB input). For each pixel it returns scores for six classes: background, hair, body skin, face skin, clothes, other. The scores are raw logits, so sepiad applies a softmax: the person mask is 1 − P(background), and the face-skin class gives portrait light its face region. OpenVINO reads `.tflite` directly. The smaller (about 250 KB) MediaPipe selfie models don't load: they use a custom operation, `Convolution2DTransposeBias`, that OpenVINO has no converter for ([ROADMAP.md](ROADMAP.md#open-questions)).
2. **Compile for the chosen device.** `core.compile_model(model, device)`. The NPU needs fixed input shapes, which this model already has. The first NPU compile takes seconds, so set `CACHE_DIR` (with `~` expanded) to make later starts near-instant.
3. **Open devices.** Read from the RGB camera by its stable path, `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5:1.0-video-index0`, and request MJPG at 1280×720, 30 fps. Turn off `exposure_dynamic_framerate` each time the camera opens (`v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`). Otherwise the camera lengthens exposure in low light and drops to about 10 fps; the setting doesn't survive a reboot and other apps can turn it back on. Do **not** use `/dev/v4l/by-id/`: the RGB and IR interfaces share one by-id name and the link points at the IR camera. Write to the v4l2loopback device (`/dev/video10`).

### Per-frame loop (30 times a second)

1. **Capture** a 1280×720 MJPG frame with OpenCV. sepiad asks for the raw JPEG and decodes it to BGR itself with `cv2.imdecode`, so decode time shows up separately from the wait for the next frame.
2. **Preprocess:** resize to 256×256 on the CPU. Converting BGR to RGB and scaling pixels from 0–255 to the range −1 to 1 are built into the model with OpenVINO's `PrePostProcessor`, so they run on the inference device. In a side-by-side preview on Oct 21, 2026 (`scripts/preview_scaling.py`), both scalings outlined the person equally well in good light, while in a dark room −1 to 1 did much better than 0 to 1. The TFLite model is already NHWC, so no layout reorder.
3. **Infer** on the chosen device. The model gives 256×256×6 class scores, and a softmax built into the model turns them into the 256×256 person mask, so the device sends back one channel instead of six. On battery, the preprocessing and the softmax on the CPU took about 10 ms per frame and pushed the NPU pipeline down to 22 fps ([BENCHMARK.md](BENCHMARK.md#blur-pipeline-power-matrix-oct-22-2026)). Inference runs while the CPU blends and sends, and each frame gets the previous frame's mask ([Staying inside the time budget](#staying-inside-the-time-budget)).
4. **Post-process the mask:** average with the previous mask to stop flicker, then shape the edge at the chosen level (`--edges`, default `medium`). `low` feathers the 256×256 mask. `medium` makes its edge twice as steep and feathers it less. `high` moves the edge onto edges in the picture with a fast guided filter: the coefficients come from a 320×180 grey copy of the frame and are applied to a 640×360 one. The blend step then scales the weights up to 720p.
5. **Apply effects:** blur, framing crop, face lighting ([OVERVIEW.md](OVERVIEW.md#feature-set)). The blur is a Gaussian on a 320×180 copy, 15×15 for `standard` and 5×5 for the lighter `portrait` (`--blur`), scaled back up to 720p.
6. **Output:** send BGR straight to the virtual camera with `pyvirtualcam` (`fmt=PixelFormat.BGR`, no extra conversion).

The background blend is one line of maths per pixel:

```latex
\text{output} = m \cdot \text{frame} + (1 - m) \cdot \text{blurred}
```

Here m is the mask value. Where m = 1 (person) the sharp frame shows; where m = 0 the blurred copy shows. `cv2.blendLinear` does this in one call without a float copy of the frame.

### Staying inside the time budget

At 30 fps each frame has about 33 ms. The CPU-side steps (MJPEG decode, blur, blend) are likely the bottleneck, not inference. sepiad times every step, so this is measured, not guessed.

- **Cheap blur:** blur a 320×180 copy, then upscale it. A small kernel on the small copy looks like a large kernel at full size.
- **Blend without float copies:** `cv2.blendLinear` instead of NumPy arithmetic.
- **iGPU post-processing:** run blur and blend on `cv2.UMat` (OpenCL) if the benchmark shows it saves CPU and power.
- **One OpenCV thread:** sepiad calls `cv2.setNumThreads(1)`. Splitting each 720p call across all 18 CPU threads cost more than it saved: on battery, one thread used 1.02 W less package power and 54% CPU instead of 129%, with blur and blend at 8.4 ms ([BENCHMARK.md](BENCHMARK.md#async-inference-and-opencv-threads-oct-22-2026)).
- **Mask from the previous frame:** the NPU segments frame N while the CPU blends frame N with the mask of frame N−1 and sends it. No frame waits for inference; the mask trails the picture by one frame, on top of the temporal smoothing. On Oct 22, 2026 this cut latency from 66.7 to 51.1 ms mean and from 86.2 to 60.9 ms at the 95th percentile, for 0.2 to 0.4 W more package power ([BENCHMARK.md](BENCHMARK.md#mask-from-the-previous-frame-plugged-in-oct-22-2026)). An earlier trial held each frame back until its own mask arrived, which added about a frame of latency.

### Running in the background

- **systemd user service:** `sepiad.service` starts the daemon and restarts it if it crashes.
- **Control without restarting:** `sepiactl` sends commands to the running daemon over a Unix socket (D-Bus can come later).
- **Camera on only during calls (v1):** run sepiad only when needed (`systemctl --user start sepiad`, or `sepiactl on/off`). Opening the camera automatically when an app reads is a later step; see [On-demand camera](#on-demand-camera-after-v1).

**Controls:** `sepiactl on|off`, `sepiactl blur standard|portrait|off`, `sepiactl edges low|medium|high`, `sepiactl background IMAGE`, `sepiactl frame on|off`, `sepiactl light on|off`, `sepiactl status`. sepiactl doesn't exist yet. For now sepiad has five flags: `--blur standard|portrait` (default `standard`), `--edges low|medium|high` (default `medium`), `--infer NPU|GPU|CPU|none` (default `NPU`; `none` passes frames through unchanged, as the no-effects baseline), `--post cpu|gpu` (default `cpu`) and `--bench SECONDS` ([BENCHMARK.md](BENCHMARK.md#harness)).

### Code

| File | What it holds |
| --- | --- |
| `sepiacam/sepiad.py` | The daemon: camera setup, the per-frame loop, `--bench` timing and power, and the bench.csv row. `build_model` adds the preprocessing and the softmax to the model; `edge_mask` shapes the mask's edge at the chosen level; `blur_background` does the blend and takes NumPy arrays or `cv2.UMat` for the iGPU |
| `tests/test_sepiad.py` | Checks that the built-in preprocessing and softmax match doing them by hand, the blend maths, and that the CPU and iGPU paths give the same image. Also that both blur modes keep the person sharp, and that higher edge levels give a narrower edge, with `high` on the picture's own edge. Run it with the venv's Python |
| `pyproject.toml` | The `sepiacam` package and the `sepiad` command; only pyvirtualcam comes from pip |
| `scripts/` | The input-scaling preview, the model-only power benchmark, and `bench_matrix.py`, which runs `sepiad --bench` over every configuration, or over those given on its command line, with idle phases in between |

## Design decisions

### Inference on the NPU

Decided Oct 22, 2026, from the model-only power benchmark ([BENCHMARK.md](BENCHMARK.md#model-only-power-oct-22-2026)). At 30 inferences per second the NPU adds 1.95 W of package power and the iGPU 5.64 W, so the NPU costs about a third as much. It is slightly slower per frame (8.7 vs 7.2 ms), but both are well inside the 33 ms budget. The CPU is ruled out because it manages only 13.8 inferences per second. With inference on the NPU, the iGPU stays free for blur and blend. `--infer GPU` and `--infer CPU` stay available for benchmarking and as fallbacks on machines without a working NPU. The full blur pipeline on battery confirmed the choice: NPU inference adds 6.19 W of package power and GPU inference 10.12 W ([BENCHMARK.md](BENCHMARK.md#blur-pipeline-power-matrix-oct-22-2026)).

### Blur and blend on the CPU, provisionally

The blur pipeline power matrix of Oct 22, 2026 came out close to a tie ([BENCHMARK.md](BENCHMARK.md#blur-pipeline-power-matrix-oct-22-2026)). With inference on the NPU, blur and blend on the iGPU cut sepiad's CPU use from 128% to 51%, and package power went from +6.19 to +6.55 W, a gap near the noise of the measurement. In that comparison the CPU cores drew about the same power either way. Both runs used OpenCV's default threads. With one thread, CPU blur costs 54% CPU and +5.00 W ([BENCHMARK.md](BENCHMARK.md#async-inference-and-opencv-threads-oct-22-2026)), and in short runs the iGPU path was worse: 59.4 against 50.7 ms mean latency, 136.7 against 56.5 ms p95, and 1.16 W more package power ([BENCHMARK.md](BENCHMARK.md#mask-from-the-previous-frame-plugged-in-oct-22-2026)). The runs had no app reading Sepia Cam, and a call changes the load on both the CPU (Chrome encodes video) and the iGPU (Chrome draws it). `--post cpu` is the default until the Meet measurement repeats both options with Meet reading Sepia Cam.

### Edge levels and blur modes

Decided Oct 23, 2026 ([BENCHMARK.md](BENCHMARK.md#blur-modes-and-edge-levels-plugged-in-oct-22-and-23-2026)). Low and medium cost the same in latency and power. Medium became the default after a live preview. High adds about 4 ms of mean latency, and its 95th percentile went over the 66 ms target (82.3 ms) at the same package power. It stays available for people who want the closest edges and accept the delay. None of the levels changed package power measurably, so they trade edge quality for latency only. Portrait blurs the same small copy as Standard with a smaller kernel and costs the same.

### On-demand camera (after v1)

With `exclusive_caps=1`, browsers list the loopback device only while a writer is streaming to it. So sepiad must keep writing to stay visible, yet it should open the real camera only while an app is reading. The plan:

1. sepiad always holds `/dev/video10` and writes a still placeholder frame (the same buffer each time, almost free) while idle.
2. It detects readers through v4l2loopback's client-usage event, if the installed version provides it, or by polling once a second for other processes holding `/dev/video10` open (`/proc/*/fd`).
3. When a reader appears, it opens the real camera; when the last reader has been gone for a few seconds, it closes it, and the camera LED goes off.

Verify the detection against Chrome: its sandboxed processes may hide their open files from `/proc`.

### Framing and lighting without a second model (v1)

Framing needs only a person box, which comes from the mask: threshold it and take `cv2.boundingRect`. Portrait light uses the model's face-skin class as the face region. Add a face detector only if these look wrong in practice.

### Latency and lip sync

sepiad measures each frame from the camera driver's buffer timestamp (CLOCK_MONOTONIC) to the moment it is sent; `--bench` reports the average and 95th percentile. The driver timestamp makes the number include time the frame spent queued before sepiad read it. The camera's own delay is about one frame period (33 ms): it needs that long to send a frame after the timestamp, and at 720p it can't go faster than 30 fps. Target: the whole number, camera delay included, stays under 66 ms (2 frames) at the 95th percentile and near 50 ms on average. For true camera-to-screen latency, film a running stopwatch through Meet's self-view once.

### Camera node selection (Level 2)

Other laptops number their cameras differently, and by-id can collide as it does here. Level 2 picks the first capture node that offers MJPG or YUYV, and skips greyscale-only (IR) nodes.

### Persistent virtual camera

`modprobe` does not survive a reboot; the module and its options go in `/etc/modules-load.d/` and `/etc/modprobe.d/` ([SETUP.md](SETUP.md), step 4), so `/dev/video10` exists before sepiad starts.

## Sources

- [MediaPipe Selfie Multiclass 256×256 model](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- [v4l2loopback](https://github.com/umlaeute/v4l2loopback)
