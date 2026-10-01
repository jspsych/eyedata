#!/bin/bash
# Executable for extract_video_metadata.sub. Runs inside the python:3.11-slim
# container in the job's scratch directory, which holds the transferred-in
# scripts, the service-account key, and the video_metadata/ output directory.
set -euo pipefail

# The container's default HOME may not be writable, and its /tmp may be too
# small for pip's build files
export HOME=$PWD
export TMPDIR=$PWD/tmp
mkdir -p "$TMPDIR"

# Install the Python dependencies into a throwaway venv in scratch.
# mediapipe is pinned to 0.10.14 because newer versions need libGLESv2, which
# the slim container lacks; it also pulls in the GUI build of OpenCV, which
# needs libGL, so it is swapped for the headless build.
python3 -m venv env
env/bin/pip install --no-cache-dir -q \
    "mediapipe==0.10.14" "av==15.1.0" "google-cloud-storage==3.9.0" \
    "pandas==2.3.3" "numpy==2.0.2"
env/bin/pip uninstall -y -q opencv-contrib-python
env/bin/pip install --no-cache-dir -q "opencv-contrib-python-headless==5.0.0.93"

mkdir -p video_metadata
env/bin/python -u metadata_from_video.py

# Keep the model and venv out of the transferred-back output
rm -rf env tmp face_landmarker.task
