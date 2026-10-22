# Architecture

How sepiad and sepiactl work, and the decisions behind the design.

## How sepiad works

sepiad runs a one-time setup, then a loop that runs on every frame. Inference runs on the device chosen by `--infer`; blur and blend run on the CPU or the iGPU, chosen by `--post`. Inference runs on the NPU by default ([Inference on the NPU](#inference-on-the-npu)). Where blur and blend run is still to be measured.

```
capture (MJPG) → preprocess → infer [NPU | GPU | CPU] → mask post-process → effects [CPU | iGPU] → virtual camera
```

### Setup (once at start)

1. **Load the model.** MediaPipe Selfie Multiclass (`selfie_multiclass_256x256.tflite`, 16 MB, 256×256 RGB input). For each pixel it returns scores for six classes: background, hair, body skin, face skin, clothes, other. The scores are raw logits, so sepiad applies a softmax: the person mask is 1 − P(background), and the face-skin class gives portrait light its face region. OpenVINO reads `.tflite` directly. The smaller (about 250 KB) MediaPipe selfie models don't load: they use a custom operation, `Convolution2DTransposeBias`, that OpenVINO has no converter for ([ROADMAP.md](ROADMAP.md#open-questions)).
2. **Compile for the chosen device.** `core.compile_model(model, device)`. The NPU needs fixed input shapes, which this model already has. The first NPU compile takes seconds, so set `CACHE_DIR` (with `~` expanded) to make later starts near-instant.
3. **Open devices.** Read from the RGB camera by its stable path, `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5:1.0-video-index0`, and request MJPG at 1280×720, 30 fps. Turn off `exposure_dynamic_framerate` each time the camera opens (`v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`). Otherwise the camera lengthens exposure in low light and drops to about 10 fps; the setting doesn't survive a reboot and other apps can turn it back on. Do **not** use `/dev/v4l/by-id/`: the RGB and IR interfaces share one by-id name and the link points at the IR camera. Write to the v4l2loopback device (`/dev/video10`).

### Per-frame loop (30 times a second)

1. **Capture** a 1280×720 MJPG frame with OpenCV (decoded to BGR on the CPU).
2. **Preprocess:** resize to 256×256, convert BGR to RGB, and scale pixels from 0–255 to the range −1 to 1. In a side-by-side preview on Oct 21, 2026 (`scripts/preview_scaling.py`), both scalings outlined the person equally well in good light, while in a dark room −1 to 1 did much better than 0 to 1. The TFLite model is already NHWC, so no layout reorder.
3. **Infer** on the chosen device: 256×256×6 class scores; a softmax turns them into the person mask.
4. **Post-process the mask:** average with the previous mask to stop flicker, upscale to 720p, feather the edges.
5. **Apply effects:** blur, framing crop, face lighting ([OVERVIEW.md](OVERVIEW.md#feature-set)).
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
- **Async inference, only if needed:** OpenVINO's `AsyncInferQueue` lets the device process frame N while the CPU blends frame N−1, but it adds one frame of latency. Use it only if the simple loop can't hold 30 fps.

### Running in the background

- **systemd user service:** `sepiad.service` starts the daemon and restarts it if it crashes.
- **Control without restarting:** `sepiactl` sends commands to the running daemon over a Unix socket (D-Bus can come later).
- **Camera on only during calls (v1):** run sepiad only when needed (`systemctl --user start sepiad`, or `sepiactl on/off`). Opening the camera automatically when an app reads is a later step; see [On-demand camera](#on-demand-camera-after-v1).

**Controls:** `sepiactl on|off`, `sepiactl blur on|off`, `sepiactl blur-strength N`, `sepiactl background IMAGE`, `sepiactl frame on|off`, `sepiactl light on|off`, `sepiactl status`.

### Core loop (sketch)

```python
import os
import cv2
import numpy as np
import openvino as ov
import pyvirtualcam

CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5:1.0-video-index0"  # RGB camera; by-id points at the IR one
DEVICE = "NPU"  # chosen by the model-only power benchmark (docs/BENCHMARK.md); GPU and CPU also work

core = ov.Core()
core.set_property({"CACHE_DIR": os.path.expanduser("~/.cache/sepia")})
model = core.compile_model("selfie_multiclass_256x256.tflite", DEVICE)

cam = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # YUYV gives only 10 fps at 720p
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 30)

prev = None
with pyvirtualcam.Camera(1280, 720, 30, device="/dev/video10", fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        inp = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1  # -1..1: much better than 0..1 in low light
        logits = model(inp[None])[0][0]                            # (256, 256, 6); runs on DEVICE
        e = np.exp(logits - logits.max(-1, keepdims=True))
        mask = 1 - e[..., 0] / e.sum(-1)                           # person = not background
        prev = mask if prev is None else 0.6 * mask + 0.4 * prev   # temporal smoothing; tune vs edge lag
        m = cv2.GaussianBlur(cv2.resize(prev, (1280, 720)), (15, 15), 0)
        blurred = cv2.resize(cv2.GaussianBlur(cv2.resize(frame, (320, 180)), (15, 15), 0), (1280, 720))
        vcam.send(cv2.blendLinear(frame, blurred, m, 1 - m))
```

## Design decisions

### Inference on the NPU

Decided Oct 22, 2026, from the model-only power benchmark ([BENCHMARK.md](BENCHMARK.md#model-only-power-oct-22-2026)). At 30 inferences per second the NPU adds 1.95 W of package power and the iGPU 5.64 W, so the NPU costs about a third as much. It is slightly slower per frame (8.7 vs 7.2 ms), but both are well inside the 33 ms budget. The CPU is ruled out because it manages only 13.8 inferences per second. With inference on the NPU, the iGPU stays free for blur and blend. `--infer GPU` and `--infer CPU` stay available for benchmarking and as fallbacks on machines without a working NPU. Revisit this if the full pipeline benchmark disagrees.

### On-demand camera (after v1)

With `exclusive_caps=1`, browsers list the loopback device only while a writer is streaming to it. So sepiad must keep writing to stay visible, yet it should open the real camera only while an app is reading. The plan:

1. sepiad always holds `/dev/video10` and writes a still placeholder frame (the same buffer each time, almost free) while idle.
2. It detects readers through v4l2loopback's client-usage event, if the installed version provides it, or by polling once a second for other processes holding `/dev/video10` open (`/proc/*/fd`).
3. When a reader appears, it opens the real camera; when the last reader has been gone for a few seconds, it closes it, and the camera LED goes off.

Verify the detection against Chrome: its sandboxed processes may hide their open files from `/proc`.

### Framing and lighting without a second model (v1)

Framing needs only a person box, which comes from the mask: threshold it and take `cv2.boundingRect`. Portrait light uses the model's face-skin class as the face region. Add a face detector only if these look wrong in practice.

### Latency and lip sync

sepiad stamps each frame when captured and when sent; `--bench` reports the average and 95th percentile. Budget: 66 ms (2 frames). For true camera-to-screen latency, film a running stopwatch through Meet's self-view once.

### Camera node selection (Level 2)

Other laptops number their cameras differently, and by-id can collide as it does here. Level 2 picks the first capture node that offers MJPG or YUYV, and skips greyscale-only (IR) nodes.

### Persistent virtual camera

`modprobe` does not survive a reboot; the module and its options go in `/etc/modules-load.d/` and `/etc/modprobe.d/` ([SETUP.md](SETUP.md), step 4), so `/dev/video10` exists before sepiad starts.

## Sources

- [MediaPipe Selfie Multiclass 256×256 model](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- [v4l2loopback](https://github.com/umlaeute/v4l2loopback)
