# Overview

What Sepia is for, what it must achieve, and what it builds.

## The problem

Linux has no system-wide camera effects, and the effects that do exist cost CPU and battery that a laptop on a video call can't spare.

- **Effects are per-app and uneven.** Google Meet in Chrome has its own blur and background replacement; WhatsApp Web has none; no browser app offers automatic framing or face lighting. Each app that does offer effects runs its own model on the CPU or GPU.
- **Nothing on Linux does what Windows Studio Effects does**: one camera, effects applied once, available to every app. Even on Windows, Microsoft now requires a Copilot+ PC with an NPU over 40 TOPS for the full set ([Microsoft](https://support.microsoft.com/en-us/windows/ai/ai-features/windows-studio-effects)).
- **The hardware for cheap inference sits unused.** The Core Ultra 5 125H has an NPU (NPU 3720, about 11 TOPS) that works on Linux through Intel's driver and OpenVINO, but no desktop app uses it for the camera. Whether it actually saves power for this workload is unknown: that is a question this project answers, not an assumption it starts from.

## Goal

One always-available virtual camera with background blur, framing and lighting that every app can select like a normal webcam, at the lowest power cost this laptop can manage, with each pipeline stage placed on NPU, iGPU or CPU by measurement.

**Secondary output:** published fps, latency and power numbers for NPU vs iGPU vs CPU on a real Linux camera workload ([BENCHMARK.md](BENCHMARK.md)). These numbers are rare and useful to others.

## Constraints

- Runs on Fedora, without a full OBS window open all day
- **Uses less power than Meet's own blur.** This is the bar the project must clear. If sepiad costs more, it still wins on WhatsApp Web and on framing/lighting, but the core claim fails.
- 30 fps at 720p, adding at most 2 frames (66 ms) of latency, so video stays in sync with audio

## Proposed solution

A background daemon, sepiad, reads the real webcam and runs a segmentation model with OpenVINO on the device chosen by benchmark. It applies the effects and writes the result to a v4l2loopback virtual camera, which browsers list as an ordinary webcam. A companion command, sepiactl, turns effects on and off while the daemon runs. Details: [ARCHITECTURE.md](ARCHITECTURE.md).

**Why a daemon and not OBS:** the OBS route works (Intel's OpenVINO plugins can target the NPU), but OBS has to stay open as a full app. sepiad is lighter, starts at login as a systemd user service, and sepiactl changes effects without restarting it.

**Alternatives considered**

| Option | Uses NPU | Effort | Notes |
| --- | --- | --- | --- |
| sepiad + sepiactl (chosen) | If it measures best | Medium | Light, always-on, fully customisable |
| Meet's built-in blur | No | None | Meet only; **the power baseline to beat** |
| [OBS + Intel OpenVINO plugins](https://github.com/intel/openvino-plugins-for-obs-studio) | Yes | Low | Includes Smart Framing; OBS must stay open |
| [fuel](https://github.com/tdv/fuel) | Yes, with a device edit | Medium | OpenVINO background removal, targets the iGPU |
| [webcam-filters](https://pypi.org/project/webcam-filters/) | No | Low | CPU-only blur |

## Feature set

Version 1 builds three camera effects (blur, framing, portrait light) on one shared segmentation pass per frame; voice focus comes free from EasyEffects; eye contact and creative filters are out of scope.

| Priority | Studio Effect | What it does | Linux implementation | Runs on | Difficulty |
| --- | --- | --- | --- | --- | --- |
| 1 | Background blur (Standard / Portrait) | Blurs everything behind you; Portrait is a stronger blur | Segmentation mask + blend; larger kernel for Portrait; image in place of blur for replacement | Measured (mask), CPU or iGPU (blur) | Easy |
| 2 | Automatic framing | Keeps you centred as you move | Person box from the mask → crop window that follows you with smoothing, then scale to 720p | Same mask, CPU (crop) | Medium |
| 3 | Portrait light | Brightens your face in poor light | Measure brightness in the face-skin region of the mask; gamma or CLAHE there, blended into the rest | Same mask, CPU (correct) | Medium |
| 4 | Voice focus | Removes background noise from the mic | EasyEffects with RNNoise or DeepFilterNet, outside sepiad | CPU (light) | Trivial |
| Skip | Eye contact (Standard / Teleprompter) | Redraws eyes to look at the camera | Needs a gaze-redirection model; no good open one ready | – | Hard |
| Skip | Creative filters | Stylised looks (illustrated, watercolour) | Style-transfer models; heavy | NPU | Hard |

**Not a Studio Effect:** whole-image exposure, white balance and contrast come from the camera. This laptop's USB camera handles them in firmware. If needed, sepiad adds a brightness-normalisation step before inference.
