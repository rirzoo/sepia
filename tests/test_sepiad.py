"""Checks for sepiad's mask and blend maths. Run: ~/.local/share/sepia/venv/bin/python tests/test_sepiad.py"""
import cv2
import numpy as np
import openvino as ov

from sepiacam.sepiad import H, MODEL, W, blur_background, build_model

# The built-in pre- and post-processing must match doing it by hand: RGB, -1..1, softmax, 1 - P(background).
core = ov.Core()
img = np.random.default_rng(0).integers(0, 256, (1, 256, 256, 3), np.uint8)
logits = core.compile_model(str(MODEL), "CPU")(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)[None] / 127.5 - 1)[0][0]
e = np.exp(logits - logits.max(-1, keepdims=True))
mask = core.compile_model(build_model(core), "CPU")(img)[0][0]
assert mask.shape == (256, 256) and np.abs(mask - (1 - e[..., 0] / e.sum(-1))).max() < 1e-4, "built-in processing"

frame = np.random.default_rng(0).integers(0, 256, (H, W, 3), np.uint8)
assert np.array_equal(blur_background(frame, np.ones((256, 256), np.float32)), frame), "person everywhere: frame unchanged"
assert not np.array_equal(blur_background(frame, np.zeros((256, 256), np.float32)), frame), "background everywhere: blurred"

cpu = blur_background(frame, mask)
gpu = blur_background(cv2.UMat(frame), cv2.UMat(mask)).get()
assert np.abs(cpu.astype(int) - gpu).max() <= 8, "iGPU and CPU paths agree"
print("ok")
