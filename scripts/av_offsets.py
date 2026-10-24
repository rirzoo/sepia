"""A/V sync test (docs/BENCHMARK.md): how far the picture lags the sound in a recording from scripts/av_sync.html.

The camera films a phone playing scripts/av_sync_video.sh's test video: once a second the screen ramps from black to
white over 100 ms, with a beep starting at the ramp's midpoint. Each event's video time is when the brightest-changing
pixels cross halfway, interpolated between frames; its sound time is when the 1 kHz tone's amplitude crosses halfway.
The script prints video minus sound for each event and their mean. Only numbers are read from the video: 64x36 grey
frames.

The phone's own screen-to-speaker offset is in every number, so compare recordings (real camera vs Sepia Cam),
not single values.

Run: ~/.local/share/sepia/venv/bin/python scripts/av_offsets.py RECORDING.webm [...]
"""
import json
import statistics
import subprocess
import sys

import numpy as np

RATE, W, H = 48000, 64, 36


def ffprobe(path, stream, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", entries,
                          "-of", "json", path], capture_output=True, check=True, text=True).stdout
    return json.loads(out)


def video_events(path):
    times = np.array([float(f["best_effort_timestamp_time"]) for f in
                      ffprobe(path, "v:0", "frame=best_effort_timestamp_time")["frames"]])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-an", "-fps_mode", "passthrough",
                          "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"], capture_output=True,
                         check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H * W).astype(np.float32)
    n = min(len(frames), len(times))
    frames, times = frames[:n], times[:n]
    screen = frames.std(0) >= np.percentile(frames.std(0), 90)  # the phone: the pixels that flash
    level = frames[:, screen].mean(1)
    half = (np.percentile(level, 5) + np.percentile(level, 95)) / 2
    events = []
    for i in np.flatnonzero((level[:-1] < half) & (level[1:] >= half)):
        f = (half - level[i]) / (level[i + 1] - level[i])  # linear between the two frames
        events.append(times[i] + f * (times[i + 1] - times[i]))
    return events


def audio_events(path):
    start = float(ffprobe(path, "a:0", "stream=start_time")["streams"][0]["start_time"])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le",
                          "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    # Room noise is as loud as the beep, so follow only the 1 kHz tone: its amplitude over 2 ms windows.
    t = np.arange(len(x)) / RATE
    box = np.ones(RATE // 500) / (RATE // 500)
    level = np.hypot(np.convolve(x * np.cos(2 * np.pi * 1000 * t), box, "same"),
                     np.convolve(x * np.sin(2 * np.pi * 1000 * t), box, "same"))
    half = (np.percentile(level, 50) + np.percentile(level, 99)) / 2  # beeps fill 5% of the time
    events = []
    for i in np.flatnonzero((level[:-1] < half) & (level[1:] >= half)):
        if not events or start + i / RATE - events[-1] > 0.5:  # beeps are 1 s apart; skip the rest of each beep
            events.append(start + i / RATE)
    return events


def offsets(path):
    sound = np.array(audio_events(path))
    pairs = [(v, sound[np.argmin(np.abs(sound - v))]) for v in video_events(path)]
    return [(v, (v - s) * 1000) for v, s in pairs if abs(v - s) < 0.4]


if __name__ == "__main__":
    for path in sys.argv[1:]:
        ms = [d for _, d in offsets(path)]
        print(f"{path}: {len(ms)} events, picture behind sound by {statistics.mean(ms):.1f} ± "
              f"{statistics.stdev(ms):.1f} ms (median {statistics.median(ms):.1f}, "
              f"range {min(ms):.1f} to {max(ms):.1f})")
