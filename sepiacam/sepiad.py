"""sepiad: read the webcam, blur the background, and write the result to the Sepia Cam virtual camera.

Run: sepiad [--model multiclass|humanseg] [--blur standard|portrait] [--edges low|medium|crisp|high]
           [--infer NPU|GPU|CPU|none] [--post cpu|gpu] [--bench SECONDS]

--bench runs for SECONDS after a short warm-up, prints the time per stage, fps, latency, CPU use and
power, and appends one row to bench.csv in the current directory (docs/BENCHMARK.md). The infer stage is
only the wait for the previous frame's mask, since inference overlaps the rest of the loop. Power needs the
RAPL counters opened first (docs/SETUP.md, step 6); without them the power columns stay empty. SIGUSR1 starts the
window again, and SIGINT during a window writes its row; scripts/bench_meet.py uses both.
"""
import argparse
import csv
import os
import signal
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
MODELS = Path(__file__).resolve().parent.parent / "models"
# file in models/ and input size (w, h). humanseg is PaddleSeg's PP-HumanSeg v2 Lite portrait model,
# converted to OpenVINO IR (docs/SETUP.md); it outputs background and person, multiclass six classes.
MODEL = {"multiclass": ("selfie_multiclass_256x256.tflite", (256, 256)),
         "humanseg": ("pp_humanseg_v2_lite/model.xml", (256, 144))}
W, H, FPS = 1280, 720, 30
RAPL = Path("/sys/class/powercap")
DOMAINS = {"pkg": "intel-rapl:0", "core": "intel-rapl:0:0", "gfx": "intel-rapl:0:1", "psys": "intel-rapl:1"}
BAT = Path("/sys/class/power_supply/BAT1")
NPU_BUSY = Path("/sys/class/accel/accel0/device/npu_busy_time_us")
EPP = Path("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")  # balance_power = battery profile
WARMUP = 30  # frames left out of --bench: OpenCL kernel builds, first inferences, camera exposure settling
STAGES = ["wait", "decode", "pre", "infer", "mask", "post", "send"]
BLURS = {"standard": (21, 21), "portrait": (5, 5)}  # Gaussian kernel on the 320x180 copy; portrait is the lighter blur


def build_model(core, model="multiclass"):
    """The segmentation model with its pre- and post-processing built in, so they run on the inference device.

    In: (1, h, w, 3) BGR uint8 at the model's input size. Out: (1, h, w) probability that each pixel is the person.
    Doing this on the CPU cost about 10 ms per frame on battery, and the device sent back all six classes.
    """
    ppp = ov.preprocess.PrePostProcessor(core.read_model(MODELS / MODEL[model][0]))
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_color_format(ColorFormat.BGR)
    ppp.input().model().set_layout(ov.Layout("NHWC" if model == "multiclass" else "NCHW"))
    # -1..1 input: much better than 0..1 in low light for multiclass (docs/ARCHITECTURE.md); humanseg was trained on it
    ppp.input().preprocess().convert_element_type(ov.Type.f32).convert_color(ColorFormat.RGB).mean(127.5).scale(127.5)
    if model == "multiclass":  # six-class logits; person = 1 - P(background)
        ppp.output().postprocess().custom(lambda logits: ops.subtract(
            np.float32(1), ops.gather(ops.softmax(logits, -1), np.int64(0), np.int64(3))).output(0))
    else:  # softmax over background and person, channels first
        ppp.output().postprocess().custom(lambda p: ops.gather(p, np.int64(1), np.int64(1)).output(0))
    return ppp.build()


def edge_mask(mask, frame, edges="crisp"):
    """Blend weights from the model's mask. low: feathered, cheapest. medium: a steeper edge. crisp (the default): a
    sharp edge at 640x360, chosen in a live preview over medium as closer to Meet's. high: the edge follows
    the frame's own edges at 640x360, through a guided filter computed at 320x180 (He and Sun, fast guided filter)."""
    if edges == "low":
        return cv2.GaussianBlur(mask, (7, 7), 0)
    if edges == "crisp":  # enlarge first, then steepen: a sharp outline where the model puts it, as in Meet
        return np.clip(cv2.resize(mask, (640, 360)) * 4 - 1.5, 0, 1)
    steep = np.clip(mask * 2 - 0.5, 0, 1)  # twice as steep around 0.5
    if edges == "medium":
        return cv2.GaussianBlur(steep, (3, 3), 0)
    grey = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)  # at 720p it cost 9 ms of latency
    i, m = cv2.resize(grey, (320, 180)).astype(np.float32), cv2.resize(steep, (320, 180))
    box = lambda x: cv2.blur(x, (5, 5))  # about 20 frame pixels across, narrow enough to follow hair
    mi, mm = box(i), box(m)
    a = (box(i * m) - mi * mm) / (box(i * i) - mi * mi + 20.0)  # eps 20 = 0.0003 in 0..1 units
    b = mm - a * mi
    # The filtered edge keeps faint tails on both sides; steepen it again (2q - 0.5, folded into a and b) and clip.
    q = cv2.add(cv2.multiply(cv2.resize(box(a) * 2, (640, 360)), grey, dtype=cv2.CV_32F),
                cv2.resize(box(b) * 2 - 0.5, (640, 360)))
    return cv2.threshold(cv2.threshold(q, 1, 1, cv2.THRESH_TRUNC)[1], 0, 0, cv2.THRESH_TOZERO)[1]


def blur_background(frame, mask, blur="standard"):
    """Blend the sharp frame over a blurred background, weighted by the mask. Takes numpy arrays, or cv2.UMat for the iGPU.

    Only the background goes into the blur, each pixel weighted by 1 - mask, so the person doesn't bleed into a halo
    around them (Meet does the same). The blur runs on a 320x180 copy."""
    w = cv2.resize(mask, (W, H))
    bg, k = cv2.subtract(1.0, cv2.resize(mask, (320, 180))), BLURS[blur]
    num = cv2.GaussianBlur(cv2.multiply(cv2.resize(frame, (320, 180)), cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR),
                                        dtype=cv2.CV_32F), k, 0)
    den = cv2.cvtColor(cv2.GaussianBlur(bg, k, 0), cv2.COLOR_GRAY2BGR)
    blurred = cv2.divide(num, den, dtype=cv2.CV_8U)  # weighted mean; 0/0 gives 0 inside the person
    return cv2.blendLinear(frame, cv2.resize(blurred, (W, H)), w, cv2.subtract(1.0, w))


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
        "model": args.model,
        "post": args.post,
        "blur": args.blur,
        "edges": args.edges,
        "battery": (BAT / "status").read_text().strip(),
        "epp": EPP.read_text().strip(),
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
           "npu_busy_pct", *(f"{k}_w" for k in DOMAINS), *(f"{s}_ms" for s in STAGES), "epp", "blur", "edges", "model"]


def append_row(row, path="bench.csv"):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        out = csv.DictWriter(f, COLUMNS, restval="")
        if new:
            out.writeheader()
        out.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Blur the webcam background and write it to the Sepia Cam virtual camera.")
    p.add_argument("--model", default="multiclass", choices=list(MODEL),
                   help="segmentation model (default multiclass); humanseg is about 3 times faster, with looser edges")
    p.add_argument("--blur", default="standard", choices=list(BLURS), help="standard, or portrait for a lighter blur")
    p.add_argument("--edges", default="crisp", choices=["low", "medium", "crisp", "high"],
                   help="edge sharpness (default crisp); high follows hair most closely and adds about 4 ms of latency")
    p.add_argument("--infer", default="NPU", choices=["NPU", "GPU", "CPU", "none"],
                   help="device that runs the model (default NPU); none passes frames through, as a baseline")
    p.add_argument("--post", default="cpu", choices=["cpu", "gpu"], help="where blur and blend run; gpu uses OpenCL")
    p.add_argument("--bench", type=float, metavar="SECONDS", help="measure for SECONDS, append a row to bench.csv, exit")
    args = p.parse_args()
    # OpenCV splits each call over every CPU thread; for 720p frames the threads mostly wait. One thread cut
    # package power by 1 W and CPU use from 129% to 54% on battery, and blur and blend still fit (docs/BENCHMARK.md).
    cv2.setNumThreads(1)

    req = None
    if args.infer != "none":
        core = ov.Core()
        core.set_property({"CACHE_DIR": os.path.expanduser("~/.cache/sepia")})  # later NPU compiles are near-instant
        req = core.compile_model(build_model(core, args.model), args.infer).create_infer_request()
    upload = cv2.UMat if args.post == "gpu" else np.asarray

    cam = open_camera()
    times, latency = {s: [] for s in STAGES}, []
    frames, start, prev, started = 0, None, None, False
    restart = []
    signal.signal(signal.SIGUSR1, lambda *_: restart.append(True))
    try:
        with pyvirtualcam.Camera(W, H, FPS, device=VCAM, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
            print(f"sepiad: {CAM} -> {vcam.device}, model {args.model}, blur {args.blur}, edges {args.edges}, infer {args.infer}, "
                  f"post {args.post}", flush=True)
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
                    inp = cv2.resize(frame, MODEL[args.model][1])
                    t.append(time.perf_counter())
                    # This frame is blended with the previous frame's mask, so no frame waits for the NPU: it
                    # segments this frame while the CPU blends, sends and decodes the next. The mask trails by a frame.
                    if started:
                        req.wait()
                    else:
                        req.infer([inp[None]])  # first frame: wait for its own mask
                    mask = req.get_output_tensor().data[0]
                    prev = mask.copy() if prev is None else 0.6 * mask + 0.4 * prev  # temporal smoothing; tune vs edge lag
                    req.start_async([inp[None]])
                    started = True
                    t.append(time.perf_counter())
                    weights = edge_mask(prev, frame, args.edges)
                    t.append(time.perf_counter())
                    out = blur_background(upload(frame), upload(weights), args.blur)
                    if isinstance(out, cv2.UMat):
                        out = out.get()
                    t.append(time.perf_counter())
                else:
                    t += [t[-1]] * 4
                vcam.send(out)
                t.append(time.perf_counter())

                if not args.bench:
                    continue
                if restart:
                    restart.clear()
                    times, latency, frames, start = {s: [] for s in STAGES}, [], 0, None
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
        if start and frames:
            report(args, start, counters(), frames, times, latency)
    finally:
        cam.release()


if __name__ == "__main__":
    main()
