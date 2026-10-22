"""sepiad: read the webcam, blur the background, and write the result to the Sepia Cam virtual camera.

Run: sepiad [--infer NPU|GPU|CPU|none] [--post cpu|gpu] [--bench SECONDS]

--bench runs for SECONDS after a short warm-up, prints the time per stage, fps, latency, CPU use and
power, and appends one row to bench.csv in the current directory (docs/BENCHMARK.md). Power needs the
RAPL counters opened first (docs/SETUP.md, step 6); without them the power columns stay empty.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pyvirtualcam
from openvino.preprocess import ColorFormat

CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5:1.0-video-index0"  # RGB camera; by-id points at the IR one
VCAM = "/dev/video10"
MODEL = Path(__file__).resolve().parent.parent / "models" / "selfie_multiclass_256x256.tflite"
W, H, FPS = 1280, 720, 30
RAPL = Path("/sys/class/powercap")
DOMAINS = {"pkg": "intel-rapl:0", "core": "intel-rapl:0:0", "gfx": "intel-rapl:0:1", "psys": "intel-rapl:1"}
BAT = Path("/sys/class/power_supply/BAT1")
NPU_BUSY = Path("/sys/class/accel/accel0/device/npu_busy_time_us")
WARMUP = 30  # frames left out of --bench: OpenCL kernel builds, first inferences, camera exposure settling
STAGES = ["wait", "decode", "pre", "infer", "mask", "post", "send"]


def build_model(core):
    """The segmentation model with its pre- and post-processing built in, so they run on the inference device.

    In: (1, 256, 256, 3) BGR uint8. Out: (1, 256, 256) probability that each pixel is the person, 1 - P(background).
    Doing this on the CPU cost about 10 ms per frame on battery, and the device sent back all six classes.
    """
    ppp = ov.preprocess.PrePostProcessor(core.read_model(MODEL))
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_color_format(ColorFormat.BGR)
    ppp.input().model().set_layout(ov.Layout("NHWC"))
    # -1..1 input: much better than 0..1 in low light (docs/ARCHITECTURE.md)
    ppp.input().preprocess().convert_element_type(ov.Type.f32).convert_color(ColorFormat.RGB).mean(127.5).scale(127.5)
    ppp.output().postprocess().custom(
        lambda logits: ops.subtract(np.float32(1), ops.gather(ops.softmax(logits, -1), np.int64(0), np.int64(3))).output(0))
    return ppp.build()


def blur_background(frame, mask):
    """Blend the sharp frame over a blurred copy, weighted by the mask. Takes numpy arrays, or cv2.UMat for the iGPU."""
    w = cv2.resize(mask, (W, H))
    blurred = cv2.resize(cv2.GaussianBlur(cv2.resize(frame, (320, 180)), (15, 15), 0), (W, H))  # small blur on a small copy
    return cv2.blendLinear(frame, blurred, w, cv2.subtract(1.0, w))


def open_camera():
    cam = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    if not cam.isOpened():
        sys.exit(f"sepiad: can't open the camera {CAM}")
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # YUYV gives only 10 fps at 720p
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cam.set(cv2.CAP_PROP_FPS, FPS)
    cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # hand over the raw MJPEG, so decode is timed on its own
    # Otherwise the camera lengthens exposure in low light and drops to about 10 fps; it resets at every reboot.
    subprocess.run(["v4l2-ctl", "-d", CAM, "-c", "exposure_dynamic_framerate=0"], check=False)
    return cam


def energy():
    try:
        return {k: int((RAPL / d / "energy_uj").read_text()) for k, d in DOMAINS.items()}
    except PermissionError:
        return None


def counters():
    return time.perf_counter(), os.times(), int(NPU_BUSY.read_text()), energy()


def report(args, start, end, frames, times, latency):
    (w0, c0, n0, e0), (w1, c1, n1, e1) = start, end
    wall = w1 - w0
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "infer": args.infer,
        "post": args.post,
        "battery": (BAT / "status").read_text().strip(),
        "seconds": round(wall, 1),
        "fps": round(frames / wall, 1),
        "latency_ms": round(float(np.mean(latency)), 1),
        "latency_p95_ms": round(float(np.percentile(latency, 95)), 1),
        "cpu_pct": round(((c1.user - c0.user) + (c1.system - c0.system)) / wall * 100),
        "npu_busy_pct": round((n1 - n0) / 1e6 / wall * 100),
    }
    for k, d in DOMAINS.items():
        top = int((RAPL / d / "max_energy_range_uj").read_text())
        row[f"{k}_w"] = round((e1[k] - e0[k]) % top / 1e6 / wall, 2) if e0 and e1 else ""  # modulo handles wrap
    for s in STAGES:
        row[f"{s}_ms"] = round(float(np.mean(times[s])) * 1000, 2)

    for k, v in row.items():
        print(f"{k:15} {v}")
    if not e0:
        print("power: RAPL counters are root-only; open them first (docs/SETUP.md, step 6)")
    append_row(row)


COLUMNS = ["time", "infer", "post", "battery", "seconds", "fps", "latency_ms", "latency_p95_ms", "cpu_pct",
           "npu_busy_pct", *(f"{k}_w" for k in DOMAINS), *(f"{s}_ms" for s in STAGES)]


def append_row(row, path="bench.csv"):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        out = csv.DictWriter(f, COLUMNS, restval="")
        if new:
            out.writeheader()
        out.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Blur the webcam background and write it to the Sepia Cam virtual camera.")
    p.add_argument("--infer", default="NPU", choices=["NPU", "GPU", "CPU", "none"],
                   help="device that runs the model (default NPU); none passes frames through, as a baseline")
    p.add_argument("--post", default="cpu", choices=["cpu", "gpu"], help="where blur and blend run; gpu uses OpenCL")
    p.add_argument("--bench", type=float, metavar="SECONDS", help="measure for SECONDS, append a row to bench.csv, exit")
    args = p.parse_args()

    req = None
    if args.infer != "none":
        core = ov.Core()
        core.set_property({"CACHE_DIR": os.path.expanduser("~/.cache/sepia")})  # later NPU compiles are near-instant
        req = core.compile_model(build_model(core), args.infer).create_infer_request()
    upload = cv2.UMat if args.post == "gpu" else np.asarray

    cam = open_camera()
    times, latency = {s: [] for s in STAGES}, []
    frames, start, prev = 0, None, None
    try:
        with pyvirtualcam.Camera(W, H, FPS, device=VCAM, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
            print(f"sepiad: {CAM} -> {vcam.device}, infer {args.infer}, post {args.post}", flush=True)
            while True:
                t = [time.perf_counter()]
                ok, buf = cam.read()
                captured = cam.get(cv2.CAP_PROP_POS_MSEC)  # driver timestamp, CLOCK_MONOTONIC in ms
                t.append(time.perf_counter())
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else None
                if frame is None:
                    continue
                t.append(time.perf_counter())
                out = frame
                if req:
                    inp = cv2.resize(frame, (256, 256))
                    t.append(time.perf_counter())
                    mask = req.infer([inp[None]])[0][0]
                    t.append(time.perf_counter())
                    prev = mask if prev is None else 0.6 * mask + 0.4 * prev  # temporal smoothing; tune vs edge lag
                    feathered = cv2.GaussianBlur(prev, (7, 7), 0)            # soft edges, cheap at 256x256
                    t.append(time.perf_counter())
                    out = blur_background(upload(frame), upload(feathered))
                    if isinstance(out, cv2.UMat):
                        out = out.get()
                    t.append(time.perf_counter())
                else:
                    t += [t[-1]] * 4
                vcam.send(out)
                t.append(time.perf_counter())

                if not args.bench:
                    continue
                frames += 1
                if not start:
                    if frames == WARMUP:
                        start, frames = counters(), 0
                else:
                    latency.append(time.clock_gettime(time.CLOCK_MONOTONIC) * 1000 - captured)
                    for s, a, b in zip(STAGES, t, t[1:]):
                        times[s].append(b - a)
                    if t[-1] - start[0] >= args.bench:
                        report(args, start, counters(), frames, times, latency)
                        return
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()


if __name__ == "__main__":
    main()
