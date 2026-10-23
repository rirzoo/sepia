"""Meet comparison (docs/BENCHMARK.md): the laptop's power in a Meet call with Meet's own blur, with no effects, and
with Meet reading Sepia Cam, blur and blend on the CPU and then on the iGPU, and with Meet reading Sepia Cam as a
passthrough (sepiad --infer none: decode and send only), which shows what the pipeline costs without the model and blur.

Run it in a terminal while you're in a Meet call in Chrome. Before each phase it says how to set Meet and waits for
Enter; after SETTLE seconds it measures for SECONDS, and you leave the laptop alone until the notification. Each
phase appends one row to bench.csv (infer = meet-blur, meet, meet+NPU with post cpu or gpu, or meet+none). Power and
cpu_pct are for the whole laptop (cpu_pct: 100 = one core); on Sepia Cam rows, fps, latency and stage times come from
sepiad over the same window. Even runs go in reverse order. Plugged in, use the battery CPU profile (Method).

Run: ~/.local/share/sepia/venv/bin/python scripts/bench_meet.py [runs, default 3] [seconds, default 60] [phase ...]
Phases are named meet-blur, meet, meet+NPU:cpu, meet+NPU:gpu and meet+none; the default is all five.
"""
import csv
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # this checkout's sepiacam, also in a worktree
from sepiacam.sepiad import BAT, DOMAINS, EPP, RAPL, append_row, energy  # noqa: E402

CSV = ROOT / "bench.csv"
SETTLE = 20  # seconds after Enter: Meet loads its blur model, Chrome's encoder settles
PHASES = [("meet-blur", "", [], "the real camera, Meet's background blur on"),  # label, post, sepiad flags, setting
          ("meet", "", [], "the real camera, Meet's effects off"),
          ("meet+NPU", "cpu", ["--post", "cpu"], "Sepia Cam, Meet's effects off"),
          ("meet+NPU", "gpu", ["--post", "gpu"], "Sepia Cam, Meet's effects off"),
          ("meet+none", "", ["--infer", "none"], "Sepia Cam, Meet's effects off")]


def busy():
    v = [int(x) for x in Path("/proc/stat").read_text().split()[1:9]]  # user nice system idle iowait irq softirq steal
    return sum(v) - v[3] - v[4]


def measure(seconds):
    e0, b0, t0 = energy(), busy(), time.perf_counter()
    time.sleep(seconds)
    e1, b1, wall = energy(), busy(), time.perf_counter() - t0
    row = {"seconds": round(wall, 1), "cpu_pct": round((b1 - b0) / os.sysconf("SC_CLK_TCK") / wall * 100)}
    for k, d in DOMAINS.items():
        top = int((RAPL / d / "max_energy_range_uj").read_text())
        row[f"{k}_w"] = round((e1[k] - e0[k]) % top / 1e6 / wall, 2)
    return row


def phase(label, post, flags, setting, seconds):
    sepiad, tmp = None, tempfile.mkdtemp(prefix="sepia-meet-")
    if flags:
        input("In Meet, turn the camera off so sepiad can open it, then press Enter. ")
        sepiad = subprocess.Popen([sys.executable, "-m", "sepiacam.sepiad", *flags, "--bench", "86400"],
                                  cwd=tmp, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.PIPE, text=True)
        print(sepiad.stdout.readline().strip(), flush=True)  # printed once Sepia Cam is streaming
    input(f"Set Meet to {setting}, camera on, then press Enter and leave the laptop alone. ")
    print(f"settling {SETTLE} s, then measuring {seconds} s", flush=True)
    time.sleep(SETTLE)
    if sepiad:
        sepiad.send_signal(signal.SIGUSR1)
    row = measure(seconds)
    if sepiad:
        sepiad.send_signal(signal.SIGINT)
        sepiad.wait()
        with open(Path(tmp) / "bench.csv", newline="") as f:
            row = {**list(csv.DictReader(f))[-1], **row}  # sepiad's fps, latency and stages; the laptop's power and CPU
    row |= {"time": datetime.now().isoformat(timespec="seconds"), "infer": label, "post": post,
            "battery": (BAT / "status").read_text().strip(), "epp": EPP.read_text().strip()}
    append_row(row, CSV)
    subprocess.run(["notify-send", "Sepia", f"{label} {post} done: {row['pkg_w']} W"], check=False)
    print(f"{label} {post}: pkg {row['pkg_w']} W, cpu {row['cpu_pct']}%", flush=True)
    return row


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if energy() is None:
        sys.exit("RAPL counters are root-only; open them first (docs/SETUP.md, step 6).")
    if EPP.read_text().strip() != "balance_power":
        print("warning: CPU not on the battery profile; run `pkexec tuned-adm profile balanced-battery`", flush=True)

    names = sys.argv[3:]
    phases = [p for p in PHASES if not names or ":".join(filter(None, p[:2])) in names]
    if names and len(phases) != len(names):
        sys.exit(f"unknown phase in {names}; phases: meet-blur, meet, meet+NPU:cpu, meet+NPU:gpu, meet+none")
    rows = {(label, post): [] for label, post, *_ in phases}
    for run in range(1, runs + 1):
        for label, post, flags, setting in phases if run % 2 else phases[::-1]:
            print(f"\nrun {run}: {label} {post}", flush=True)
            rows[(label, post)].append(phase(label, post, flags, setting, seconds))

    def cell(values):
        sd = statistics.stdev(values) if len(values) > 1 else 0
        return f"{statistics.mean(values):.2f} ± {sd:.2f}"

    base = rows.get(("meet", ""), [])  # no meet phase: the over-meet column stays empty
    print("\n| phase | pkg W | over meet W | core W | gfx W | laptop cpu % | fps | latency ms | p95 ms |")
    print("|" + " --- |" * 9)
    for (label, post), rs in rows.items():
        over = [float(r["pkg_w"]) - float(b["pkg_w"]) for r, b in zip(rs, base)]
        sepia = [f"{statistics.mean(float(r[k]) for r in rs):.1f}" if "fps" in rs[0] else "" for k in
                 ["fps", "latency_ms", "latency_p95_ms"]]
        print(f"| {label} {post} | {cell([float(r['pkg_w']) for r in rs])} | {cell(over) if over else ""} | "
              f"{cell([float(r['core_w']) for r in rs])} | {cell([float(r['gfx_w']) for r in rs])} | "
              f"{statistics.mean(r['cpu_pct'] for r in rs):.0f} | " + " | ".join(sepia) + " |")


if __name__ == "__main__":
    main()
