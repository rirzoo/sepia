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

Setup is complete and checked end to end on this laptop: real 720p frames → model on the NPU → the "Sepia Cam" virtual camera → read back by another process. A model-only power benchmark put inference on the NPU, which uses about a third of the iGPU's power. The blur-only sepiad works. On battery, with the full pipeline, NPU inference adds 6.19 W of package power against 10.12 W for the iGPU, so the NPU stays the default. Running OpenCV on one thread saved another 1.02 W, and with one thread blur on the CPU beats the iGPU on both latency and power. Blending each frame with the previous frame's mask brought latency to 51.1 ms on average and 60.9 ms at the 95th percentile, at 29.7 fps, against a 66 ms budget. Blur comes in Standard and a lighter Portrait, with edges at low, medium, crisp (the default) or high. In a Meet call, Meet's own blur costs about 1 W less than Sepia Cam, and Sepia's p95 latency rises to 78 to 84 ms. The blur now uses only background pixels, as Meet's does, so the person has no halo. With no model and no blur, Sepia Cam adds only 0.30 W in a call, so the model and the blur are where the power goes. A smaller model, PP-HumanSeg v2 Lite, is the fallback for lower power or a laptop without a working NPU or iGPU; in a call it costs about what Meet's blur does, 1.11 W less than the default model. The mask now holds still while the person does, as Meet's does. Chrome on Linux can't correct for Sepia's delay, so it has to stay small. A recording test in Chrome found no large lip-sync offset from Sepia. Next: keep its delay within one frame at the 95th percentile in a call. See [ROADMAP.md](docs/ROADMAP.md#next-steps).

## Docs

| Doc | What it covers |
| --- | --- |
| [OVERVIEW.md](docs/OVERVIEW.md) | The problem, goal, constraints, alternatives considered, and the feature set |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How sepiad works: the pipeline, the code map, sepiactl, and design decisions |
| [SETUP.md](docs/SETUP.md) | This laptop's hardware and the Fedora setup steps, including the known workarounds |
| [BENCHMARK.md](docs/BENCHMARK.md) | How devices are compared (baselines, matrix, metrics, method) and the results so far |
| [ROADMAP.md](docs/ROADMAP.md) | Levels 1–4, risks, open questions and next steps |
