# Setup

This laptop's hardware and how to set Fedora up for Sepia, including the workarounds found so far.

## This laptop (checked Oct 21, 2026)

| Part | Detail |
| --- | --- |
| CPU | Intel Core Ultra 5 125H (Meteor Lake) |
| NPU | NPU 3720, `/dev/accel/accel0` (mode `0666`, no group membership needed), `intel_vpu` kernel driver loaded |
| GPU | Integrated Intel Arc graphics only, no discrete GPU |
| Camera | USB UVC, Chicony `04f2:b7e8`. **720p30 only in MJPG**; YUYV at 720p is 10 fps. With `exposure_dynamic_framerate=1` it drops to about 10 fps in low light |
| Camera nodes | `/dev/video0` RGB camera, `/dev/video2` IR (Windows Hello) camera. Use `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5:1.0-video-index0`; by-id points at the IR camera |
| Virtual camera | `/dev/video10`, "Sepia Cam" (v4l2loopback 0.15.4) |
| Kernel | 7.2.11 |
| Python | 3.14 (system); venv at `~/.local/share/sepia/venv` with system packages visible |
| Fedora packages | `openvino` / `openvino-plugins` / `python3-openvino` / `intel-npu-compiler` 2025.1, `intel-npu-driver` 1.16.0 (pinned; see step 2), `python3-opencv` 4.13, pyvirtualcam 0.15.0 (pip, in the venv) |
| Secure Boot | Disabled, so the v4l2loopback akmod needs no signing |

## Prerequisites

Four things must work before sepiad: the camera, the NPU driver, OpenVINO seeing the NPU and GPU, and a virtual camera device.

1. **Camera works.** Already confirmed: USB UVC at `/dev/video0`, 720p30 in MJPG. Check with `v4l2-ctl -d /dev/video0 --list-formats-ext`.
2. **NPU, GPU and Python packages,** all from Fedora's repos (Fedora's packages match the system Python 3.14, so no pip wheels for these):

   ```bash
   sudo dnf install intel-npu-driver intel-npu-compiler openvino openvino-plugins python3-openvino \
                    intel-compute-runtime python3-opencv python3-numpy
   sudo dnf downgrade intel-npu-driver-1.16.0-3.fc44   # 1.32.0 can't compile with the 2025.1 compiler
   sudo dnf versionlock add intel-npu-driver
   ```

   `intel-npu-compiler` is a separate package; without it the NPU can't compile any model (`ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`). Fedora 44 shipped `intel-npu-driver` 1.16.0 with OpenVINO 2025.1. A later update moved the driver to 1.32.0 but left OpenVINO and the compiler at 2025.1, and every NPU compile then fails (`Failed to create compiler executable`). So the driver is downgraded and pinned until Fedora updates OpenVINO; then remove the lock with `sudo dnf versionlock delete intel-npu-driver`. `intel-compute-runtime` provides OpenCL for the iGPU, needed by OpenVINO's GPU plugin and by `cv2.UMat`. No `render` group needed here: `/dev/accel/accel0` is already world-accessible. Fallback driver: the [xanderlent/intel-npu-driver COPR](https://copr.fedorainfracloud.org/coprs/xanderlent/intel-npu-driver).
3. **OpenVINO sees all three devices; OpenCV sees OpenCL.** `pyvirtualcam` is the only pip package; install it in a venv that can still see the system packages:

   ```bash
   python3 -c "import openvino as ov; print(ov.Core().available_devices)"   # expect ['CPU', 'GPU', 'NPU']
   python3 -c "import cv2; print(cv2.ocl.haveOpenCL())"                     # expect True
   python3 -m venv --system-site-packages ~/.local/share/sepia/venv
   ~/.local/share/sepia/venv/bin/pip install pyvirtualcam
   ```

   Download the model into `models/`:

   ```bash
   curl -LO https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
   ```
4. **Virtual camera.** v4l2loopback comes from RPM Fusion free. Install the release package from `download1.rpmfusion.org`: the `mirrors.rpmfusion.org` redirector sent this laptop to a mirror that timed out. The akmod builds the kernel module on install and rebuilds it at boot after kernel updates. `exclusive_caps=1` is required, or Chrome and Firefox won't list the device. Make it load at every boot:

   ```bash
   sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
   sudo dnf install akmod-v4l2loopback
   echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
   echo 'options v4l2loopback devices=1 video_nr=10 card_label="Sepia Cam" exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
   sudo modprobe v4l2loopback
   ```
5. **Voice focus (optional).** `sudo dnf install easyeffects`, then enable its noise-reduction plugin.

## Sources

- [Fedora package review: intel-npu-driver](https://bugzilla.redhat.com/show_bug.cgi?id=2463156)
- [xanderlent/intel-npu-driver COPR](https://copr.fedorainfracloud.org/coprs/xanderlent/intel-npu-driver)
