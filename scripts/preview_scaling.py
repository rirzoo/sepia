"""Live side-by-side preview of the person mask for two input scalings, 0..1 and -1..1.

The background (where the model sees no person) is tinted green; pick the side whose
green edge follows you most closely. Nothing is saved.

Run: ~/.local/share/sepia/venv/bin/python scripts/preview_scaling.py [NPU|GPU|CPU]
Quit: q, Esc, or close the window.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5:1.0-video-index0"  # RGB camera; by-id points at the IR one
MODEL = Path(__file__).resolve().parent.parent / "models" / "selfie_multiclass_256x256.tflite"
SCALINGS = {"0..1": lambda x: x / 255, "-1..1": lambda x: x / 127.5 - 1}
GREEN = np.array([0, 160, 0], np.float32)
PANEL = (640, 360)
WINDOW = "Sepia: input scaling (q to quit)"


def person_mask(model, rgb, scale):
    logits = model(scale(rgb)[None])[0][0]                         # (256, 256, 6)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return 1 - e[..., 0] / e.sum(-1)                               # person = not background


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "NPU"
    core = ov.Core()
    core.set_property({"CACHE_DIR": os.path.expanduser("~/.cache/sepia")})
    model = core.compile_model(str(MODEL), device)

    subprocess.run(["v4l2-ctl", "-d", CAM, "-c", "exposure_dynamic_framerate=0"], check=False)
    cam = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow(WINDOW)
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            small = cv2.resize(frame, PANEL).astype(np.float32)
            rgb = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32)
            panels = []
            for name, scale in SCALINGS.items():
                t = time.perf_counter()
                m = person_mask(model, rgb, scale)
                ms = (time.perf_counter() - t) * 1000
                m = cv2.resize(m, PANEL)[..., None]
                panel = (small * m + (0.4 * small + 0.6 * GREEN) * (1 - m)).astype(np.uint8)
                label = f"{name}  {device} {ms:.1f} ms  person {m.mean() * 100:.0f}%"
                cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(panel)
            cv2.imshow(WINDOW, np.hstack(panels))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
