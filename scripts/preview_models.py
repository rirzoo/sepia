"""Live side-by-side preview of the person mask from the 16 MB multiclass model (-1..1 input, left) and a smaller
model (right): Meet's 250 KB meet_1655 (144x256 input), ML Kit's 250 KB 256x256 selfie model, or PaddleSeg's
PP-HumanSeg v2 Lite portrait model (144x256 input, OpenVINO IR converted from the Paddle download, docs/SETUP.md).

The background (where the model sees no person) is tinted green; compare how closely each green edge follows you.
Press m to switch the right-hand model, s to switch its input scaling between 0..1 and -1..1 (each model starts at
its own). Nothing is saved. The window can be resized.

Run: PYTHONPATH=<worktree> ~/.local/share/sepia/venv/bin/python scripts/preview_models.py [NPU|GPU|CPU]
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
import openvino.opset13 as ops
from openvino.frontend import ConversionExtension

CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5:1.0-video-index0"  # RGB camera; by-id points at the IR one
MODELS = Path(__file__).resolve().parent.parent / "models"
SCALINGS = {"0..1": lambda x: x / 255, "-1..1": lambda x: x / 127.5 - 1}
GREEN = np.array([0, 160, 0], np.float32)
SMALL = {"meet_1655": ("selfie_segmentation_landscape.tflite", "0..1"), "mlkit 256x256": ("selfie_segmenter.tflite", "0..1"),
         "pp-humanseg v2": ("pp_humanseg_v2_lite/model.xml", "-1..1")}  # file in models/, input scaling
PANEL = (960, 540)
WINDOW = "Sepia: multiclass vs small model (m: model, s: scaling, q: quit)"


def transpose_conv_bias(node):
    """MediaPipe's Convolution2DTransposeBias as a standard transposed convolution plus bias (stride 2, no padding:
    its only use in the selfie models is a 2x2 kernel that doubles the last feature map)."""
    x, w, b = (node.get_input(i) for i in range(3))
    y = ops.convolution_backprop_data(ops.transpose(x, [0, 3, 1, 2]), ops.transpose(w, [3, 0, 1, 2]), strides=[2, 2],
                                      pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1])
    return [ops.add(ops.transpose(y, [0, 2, 3, 1]), b).output(0)]


def multiclass_mask(model, frame):
    rgb = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32)
    logits = model(SCALINGS["-1..1"](rgb)[None])[0][0]           # (256, 256, 6)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return 1 - e[..., 0] / e.sum(-1)                               # person = not background


def small_mask(model, frame, scale):
    shape = model.input().get_shape()
    nchw = shape[1] == 3                                           # PP-HumanSeg; the MediaPipe models are NHWC
    h, w = shape[2:4] if nchw else shape[1:3]
    rgb = scale(cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32))
    if nchw:
        return model(rgb.transpose(2, 0, 1)[None])[0][0, 1]        # (h, w), softmax: channel 1 is the person
    return model(rgb[None])[0][0, ..., 0]                          # (h, w), sigmoid: person probability


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "NPU"
    core = ov.Core()
    core.set_property({"CACHE_DIR": os.path.expanduser("~/.cache/sepia")})
    core.add_extension(ConversionExtension("Convolution2DTransposeBias", transpose_conv_bias))
    multiclass = core.compile_model(str(MODELS / "selfie_multiclass_256x256.tflite"), device)
    small = {name: core.compile_model(str(MODELS / f), device) for name, (f, _) in SMALL.items()}

    subprocess.run(["v4l2-ctl", "-d", CAM, "-c", "exposure_dynamic_framerate=0"], check=False)
    cam = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)

    scaling, name = "0..1", "meet_1655"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 2 * PANEL[0], PANEL[1])
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            shown = cv2.resize(frame, PANEL).astype(np.float32)
            panels = []
            for label, run in [("multiclass", lambda: multiclass_mask(multiclass, frame)),
                               (f"{name} {scaling}", lambda: small_mask(small[name], frame, SCALINGS[scaling]))]:
                t = time.perf_counter()
                m = run()
                ms = (time.perf_counter() - t) * 1000
                m = cv2.resize(m, PANEL)[..., None]
                panel = (shown * m + (0.4 * shown + 0.6 * GREEN) * (1 - m)).astype(np.uint8)
                text = f"{label}  {device} {ms:.1f} ms  person {m.mean() * 100:.0f}%"
                cv2.putText(panel, text, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                panels.append(panel)
            cv2.imshow(WINDOW, np.hstack(panels))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                names = list(SMALL)
                name = names[(names.index(name) + 1) % len(names)]
                scaling = SMALL[name][1]
            if key == ord("s"):
                scaling = "-1..1" if scaling == "0..1" else "0..1"
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
