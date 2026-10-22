"""Model-only power benchmark: run inference at 30 per second on each device and measure power.

Power comes from the chip's RAPL energy counters (package, CPU cores, iGPU, and the platform),
with battery power as a whole-system cross-check when unplugged. The NPU has no RAPL domain of
its own; it is inside "rest" (package minus cores minus iGPU).

Camera off; random input. Idle phases alternate with device phases (CPU last, as it heats the
chip most), and each device is compared with the mean of the idle phases on either side of it,
so slow drift in background load cancels out. The spread of the idle phases shows how noisy
the run was.

The RAPL counters are root-only; open them until the next reboot with
  pkexec chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj /sys/class/powercap/intel-rapl:0:0/energy_uj \
                   /sys/class/powercap/intel-rapl:0:1/energy_uj /sys/class/powercap/intel-rapl:1/energy_uj

Run: ~/.local/share/sepia/venv/bin/python scripts/bench_infer_power.py [seconds per phase, default 60]
"""
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov

MODEL = Path(__file__).resolve().parent.parent / "models" / "selfie_multiclass_256x256.tflite"
RAPL = Path("/sys/class/powercap")
DOMAINS = {"pkg": "intel-rapl:0", "core": "intel-rapl:0:0", "gfx": "intel-rapl:0:1", "psys": "intel-rapl:1"}
BAT = Path("/sys/class/power_supply/BAT1")
NPU_BUSY = Path("/sys/class/accel/accel0/device/npu_busy_time_us")
RATE = 30  # inferences per second, as in sepiad
SETTLE = 5  # seconds after compiling before measuring
COOLDOWN = 15  # seconds before each idle phase; power stays high for a while after load
COLS = ["pkg", "core", "gfx", "rest", "psys", "bat"]


def energy():
    return {k: int((RAPL / d / "energy_uj").read_text()) for k, d in DOMAINS.items()}


MAX = {k: int((RAPL / d / "max_energy_range_uj").read_text()) for k, d in DOMAINS.items()}


def phase(seconds, work=None):
    """Run `work(stop_event)` for `seconds`; return mean watts per domain (and battery, if discharging)."""
    on_battery = (BAT / "status").read_text().strip() == "Discharging"
    stop, bat = threading.Event(), []
    worker = threading.Thread(target=work, args=(stop,)) if work else None
    e0, t0 = energy(), time.monotonic()
    if worker:
        worker.start()
    while time.monotonic() - t0 < seconds:
        time.sleep(1)
        if on_battery:
            bat.append(int((BAT / "power_now").read_text()) / 1e6)
    e1, t1 = energy(), time.monotonic()
    stop.set()
    if worker:
        worker.join()
    w = {k: ((e1[k] - e0[k]) % MAX[k]) / 1e6 / (t1 - t0) for k in DOMAINS}  # modulo handles counter wrap
    w["rest"] = w["pkg"] - w["core"] - w["gfx"]
    w["bat"] = sum(bat) / len(bat) if bat else float("nan")
    return w


def fmt(w):
    return " ".join(f"{w[c]:6.2f}" for c in COLS)


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    try:
        energy()
    except PermissionError:
        sys.exit("RAPL counters are root-only; open them first (see the command at the top of this file).")

    core = ov.Core()
    core.set_property({"CACHE_DIR": os.path.expanduser("~/.cache/sepia")})
    x = np.random.rand(1, 256, 256, 3).astype(np.float32) * 2 - 1

    print(f"idle {seconds} s ...", flush=True)
    idles, rows = [phase(seconds)], []
    for device in ["NPU", "GPU", "CPU"]:
        req = core.compile_model(str(MODEL), device).create_infer_request()
        for _ in range(20):
            req.infer([x])
        time.sleep(SETTLE)
        count, stats = [0], {}

        def run(stop):
            c0, b0, w0 = os.times(), int(NPU_BUSY.read_text()), time.perf_counter()
            nxt = w0
            while not stop.is_set():
                req.infer([x])
                count[0] += 1
                nxt += 1 / RATE
                delay = nxt - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    nxt = time.perf_counter()  # can't keep up (CPU); run flat out
            c1, wall = os.times(), time.perf_counter() - w0
            stats.update(rate=count[0] / wall, cpu=((c1.user - c0.user) + (c1.system - c0.system)) / wall * 100,
                         npu=(int(NPU_BUSY.read_text()) - b0) / 1e6 / wall * 100)

        print(f"{device} {seconds} s ...", flush=True)
        rows.append((device, phase(seconds, run), stats))
        del req
        time.sleep(COOLDOWN)
        print(f"idle {seconds} s ...", flush=True)
        idles.append(phase(seconds))

    print(f"\nWatts       {' '.join(f'{c:>6}' for c in COLS)}")
    for i, w in enumerate(idles):
        print(f"idle {i + 1}      {fmt(w)}")
    spread = max(w["pkg"] for w in idles) - min(w["pkg"] for w in idles)
    print(f"idle package spread: {spread:.2f} W (device differences smaller than this are noise)")

    print(f"\nOver idle   {' '.join(f'{c:>6}' for c in COLS)}  inf/s  mJ/inf(pkg)  CPU %  NPU busy %")
    for i, (device, w, s) in enumerate(rows):
        extra = {c: w[c] - (idles[i][c] + idles[i + 1][c]) / 2 for c in COLS}
        print(f"{device:11} {fmt(extra)}  {s['rate']:5.1f}  {extra['pkg'] / s['rate'] * 1000:11.1f}"
              f"  {s['cpu']:5.0f}  {s['npu']:10.0f}")


if __name__ == "__main__":
    main()
