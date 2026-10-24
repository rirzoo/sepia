#!/bin/bash
# Test video for the A/V sync test (scripts/av_offsets.py): 60 s, portrait, for a phone. Once a second the screen
# ramps from black to white over 100 ms, holds for 100 ms and goes black; a 1 kHz beep starts at the ramp's midpoint.
# Run: scripts/av_sync_video.sh [OUT, default av_sync_test.mp4]
ffmpeg -v error -y \
  -f lavfi -i "color=black:s=720x1280:r=60:d=60,format=gray,geq=lum='255*if(lt(mod(T,1),0.2),min(mod(T,1)/0.1,1),0)'" \
  -f lavfi -i "aevalsrc='if(between(mod(t,1),0.05,0.1),0.8*sin(2*PI*1000*t),0)':s=48000:d=60" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 128k "${1:-av_sync_test.mp4}"
