# Sepia: Low-Power Camera Effects on Fedora (NPU vs iGPU vs CPU)

Oct 21, 2026 · @Romir Zadoo

**Sepia** is a Linux camera-effects tool: one virtual camera with background blur, framing and lighting that every app can use, at the lowest CPU and battery cost this laptop allows. Each stage runs on whichever device (NPU, integrated GPU or CPU) measures best, and the measurements are published as a secondary output. It is named after the cuttlefish genus *Sepia*: cuttlefish change how their surroundings look around them, as Sepia does with the background behind you. It joins *homer* and *mus* in the animal-named projects.

| Name | What it is |
| --- | --- |
| Sepia | The project |
| `sepiacam` | The package (`pip install sepiacam`, `import sepiacam`) |
| `sepiad` | The background daemon that runs the camera pipeline, as a systemd user service; `sepiad --bench` produces the benchmark data |
| `sepiactl` | The command that controls the running daemon (`sepiactl blur on`) |

## Status

Setup is complete and checked end to end on this laptop: real 720p frames → model on the NPU → the "Sepia Cam" virtual camera → read back by another process. Next: write the blur-only sepiad with `--bench`. See [ROADMAP.md](docs/ROADMAP.md#next-steps).

## Docs

| Doc | What it covers |
| --- | --- |
| [OVERVIEW.md](docs/OVERVIEW.md) | The problem, goal, constraints, alternatives considered, and the feature set |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How sepiad works: the pipeline, the core loop sketch, sepiactl, and design decisions |
| [SETUP.md](docs/SETUP.md) | This laptop's hardware and the Fedora setup steps, including the known workarounds |
| [BENCHMARK.md](docs/BENCHMARK.md) | How devices are compared (baselines, matrix, metrics, method) and the results so far |
| [ROADMAP.md](docs/ROADMAP.md) | Levels 1–4, risks, open questions and next steps |
