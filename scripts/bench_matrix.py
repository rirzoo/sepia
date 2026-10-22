"""Full-pipeline power benchmark: idle phases alternating with `sepiad --bench` runs (docs/BENCHMARK.md, Method).

Each configuration is compared with the mean of the idle phases on either side of it. Every phase, idle ones
included (infer = idle), is appended to bench.csv in the repo root; the summary at the end is mean ± standard
deviation over the runs. Camera on during the sepiad phases. Needs the RAPL counters open (docs/SETUP.md, step 6).

Run it under systemd-inhibit, or the laptop may suspend during an idle phase:
  systemd-inhibit --what=sleep:idle ~/.local/share/sepia/venv/bin/python scripts/bench_matrix.py [runs] [seconds] [config ...]
Runs default to 3 and seconds per phase to 60. Each config is a string of sepiad flags, run from this checkout as
`python -m sepiacam.sepiad`, or a .py path and its flags to run another sepiad copy (an A/B against an older
commit). Without configs it runs the 7 below. Plugged in, use the battery CPU profile (docs/BENCHMARK.md, Method).
In a worktree, set PYTHONPATH to it, or the editable install imports the main checkout's sepiacam.
"""
import csv
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from sepiacam.sepiad import BAT, DOMAINS, EPP, RAPL, append_row, energy

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "bench.csv"
CONFIGS = [f"--infer {i} --post {p}" for i, p in [("none", "cpu"), ("NPU", "cpu"), ("NPU", "gpu"), ("GPU", "cpu"),
                                                  ("GPU", "gpu"), ("CPU", "cpu"), ("CPU", "gpu")]]  # CPU inference last: it heats the chip most
COOLDOWN = 15  # seconds before each idle phase; power stays high for a while after load
POWER = ["pkg", "core", "gfx", "rest", "psys"]
OTHER = ["fps", "latency_ms", "latency_p95_ms", "cpu_pct", "npu_busy_pct"]


def watts(row):
    w = {k: float(row[f"{k}_w"]) for k in DOMAINS}
    w["rest"] = w["pkg"] - w["core"] - w["gfx"]  # includes the NPU, which has no RAPL domain
    return w


def idle(seconds):
    time.sleep(COOLDOWN)
    e0, t0 = energy(), time.perf_counter()
    time.sleep(seconds)
    e1, wall = energy(), time.perf_counter() - t0
    row = {"time": datetime.now().isoformat(timespec="seconds"), "infer": "idle",
           "battery": (BAT / "status").read_text().strip(), "epp": EPP.read_text().strip(), "seconds": round(wall, 1)}
    for k, d in DOMAINS.items():
        top = int((RAPL / d / "max_energy_range_uj").read_text())
        row[f"{k}_w"] = round((e1[k] - e0[k]) % top / 1e6 / wall, 2)
    append_row(row, CSV)
    return row


def last_row():
    with open(CSV, newline="") as f:
        return list(csv.DictReader(f))[-1]


def command(config):
    args = shlex.split(config)
    return [sys.executable, *args] if args[0].endswith(".py") else [sys.executable, "-m", "sepiacam.sepiad", *args]


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    configs = sys.argv[3:] or CONFIGS
    if energy() is None:
        sys.exit("RAPL counters are root-only; open them first (docs/SETUP.md, step 6).")
    if EPP.read_text().strip() != "balance_power":
        print("warning: CPU not on the battery profile; run `pkexec tuned-adm profile balanced-battery`", flush=True)

    results = {c: [] for c in configs}  # config -> per-run dict of extra watts and other metrics
    for run in range(1, runs + 1):
        print(f"run {run}: idle", flush=True)
        before = idle(seconds)
        for config in configs:
            print(f"run {run}: {config}", flush=True)
            subprocess.run([*command(config), "--bench", str(seconds)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            row = last_row()
            print(f"run {run}: idle", flush=True)
            after = idle(seconds)
            base = {k: (watts(before)[k] + watts(after)[k]) / 2 for k in POWER}
            results[config].append({**{k: watts(row)[k] - base[k] for k in POWER},
                                           **{k: float(row[k]) for k in OTHER}})
            before = after

    def cell(values, digits=2):
        sd = statistics.stdev(values) if len(values) > 1 else 0
        return f"{statistics.mean(values):+.{digits}f} ± {sd:.{digits}f}"

    print("\n| config | " + " | ".join(f"{k} W over idle" for k in POWER) + " | " + " | ".join(OTHER) + " |")
    print("|" + " --- |" * (1 + len(POWER) + len(OTHER)))
    for config, rs in results.items():
        power = [cell([r[k] for r in rs]) for k in POWER]
        other = [f"{statistics.mean(r[k] for r in rs):.1f}" for k in OTHER]
        print(f"| `{config}` | " + " | ".join(power) + " | " + " | ".join(other) + " |")


if __name__ == "__main__":
    main()
