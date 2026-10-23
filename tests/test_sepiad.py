"""Checks for sepiad's mask and blend maths. Run: ~/.local/share/sepia/venv/bin/python tests/test_sepiad.py"""
import cv2
import numpy as np
import openvino as ov

from sepiacam.sepiad import BLURS, H, MODEL, W, blur_background, build_model, edge_mask

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
for blur in BLURS:
    assert np.array_equal(blur_background(frame, np.ones((256, 256), np.float32), blur), frame), f"{blur}: person kept"
assert not np.array_equal(*(blur_background(frame, np.zeros((256, 256), np.float32), b) for b in BLURS)), "modes differ"

# A dark person on the left of a bright background, and a coarse mask of them: higher edge levels give a narrower
# transition from person to background, and high puts it at the frame's own edge (x = 640).
scene = np.full((H, W, 3), 200, np.uint8)
scene[:, :640] = 60
coarse = np.zeros((256, 256), np.float32)
coarse[:, :128] = 1
coarse = cv2.GaussianBlur(coarse, (15, 15), 0)
width = {}
for edges in ("low", "medium", "high"):
    row = cv2.resize(edge_mask(coarse, scene, edges), (W, H))[H // 2]
    assert -0.05 < row.min() and row.max() < 1.05, f"{edges}: weights in range"
    width[edges] = int(((row > 0.1) & (row < 0.9)).sum())
assert width["low"] > width["medium"] > width["high"], f"sharper edges are narrower: {width}"
assert abs(int(np.argmax(cv2.resize(edge_mask(coarse, scene, "high"), (W, H))[H // 2] < 0.5)) - 640) <= 3, "high: at x = 640"
step = np.zeros((256, 256), np.float32)
step[:, :128] = 1
for blur in BLURS:  # the dark person must not bleed into the bright background next to them
    assert blur_background(scene, cv2.GaussianBlur(step, (9, 9), 0), blur)[H // 2, 650:700].min() >= 185, f"{blur}: no halo"
print("ok")
