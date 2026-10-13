# Extracts per-frame eye-region and head-pose metadata from the eyedata24
# webcam webm clips stored on Google Cloud.
#
# Every frame is run through the MediaPipe Face Landmarker, and the left and
# right eye patches are cut out with et_util.process_functions (eyetracking-utils).
# For every frame of every video, records:
#   video_id, subject_id, frame_idx, dataset, face_detected,
#   left_eye_brightness       mean grayscale value of the left eye patch (0-255)
#   left_right_brightness_diff  left eye patch brightness minus right eye patch brightness
#   left_eye_brightness_sd    SD of grayscale values within the left eye patch
#   interocular_distance_px   distance between the iris centers, in pixels
#   interocular_distance_norm interocular distance as a fraction of frame width
#   head_roll, head_pitch, head_yaw  degrees, from position_utils.calculate_head_pose
#   eye_aspect_ratio          max of the two eyes' EAR (lid gap / corner-to-corner width)
#   sharpness                 SD of the Laplacian over the left eye patch
#   sensor_noise              Immerkaer noise estimate (sigma) over the left eye patch
# Metrics are NaN on frames with no face detected (or an unusably small eye patch).
#
# "Left" follows eyetracking-utils' getLeftEye, i.e. the eye on the left of the image.
#
# Subjects listed in include_list.csv are written to the public CSV; all
# others go to the private CSV. Output goes to <this dir>/eyedata24_metadata.
# The .tar files are streamed from the bucket and the videos are decoded in
# memory, so nothing but the CSVs is written to disk.
# The script is resumable: videos already present in the output CSVs are skipped.
#
# Runs on CHTC via eyedata24_extract_videos_metadata.sub / .sh.

#%%
import io
import json
import math
import os
import sys
import tarfile
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import av
import cv2
import google.auth
import mediapipe as mp
import numpy as np
import pandas as pd
from google.cloud import storage

#%%
current_dir = Path(__file__).parent.resolve()

sys.path.insert(0, str(current_dir))
import blink_utils
import position_utils
# process_functions.py comes from eyetracking-utils/et_util. The CHTC job transfers
# it in next to this script; locally, fall back to the sibling eyetracking-utils checkout.
sys.path.append(str(current_dir.parent.parent / "eyetracking-utils" / "et_util"))
from process_functions import getLeftEye, getRightEye

#%%
#### CONFIG ###########

# ---------------------------------------------------------------------------
# GOOGLE ACCOUNT INFO
# Authentication uses a service-account key file (.json), found in this order:
#   1. the GOOGLE_APPLICATION_CREDENTIALS environment variable
#      (the CHTC job sets this to the key it transfers in)
#   2. ~/.config/gcloud-keys/sa-key.json
# If neither exists, falls back to application-default credentials from
# `gcloud auth application-default login`.
# GCP_PROJECT_ID: the Google Cloud project that owns the buckets.
DEFAULT_KEY_PATH = Path.home() / ".config" / "gcloud-keys" / "sa-key.json"
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or (
    str(DEFAULT_KEY_PATH) if DEFAULT_KEY_PATH.exists() else None
)
GCP_PROJECT_ID = "ardent-time-400616"
# ---------------------------------------------------------------------------

INCLUDE_LIST_BUCKET = "webcam-jspsych-frames"
INCLUDE_LIST_BLOB = "v1/include_list.csv"

VIDEO_BUCKET = "webcam-jspsych-webm"
DATASET = "eyedata24"

MODEL_PATH = current_dir / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

OUTPUT_DIR = current_dir / "eyedata24_metadata"
PUBLIC_CSV = OUTPUT_DIR / "eyedata24_public_videos_metadata.csv"
PRIVATE_CSV = OUTPUT_DIR / "eyedata24_private_videos_metadata.csv"

# On CHTC, os.cpu_count() reports the whole machine, so the job passes its CPU request in
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
# Caps how many videos are held in RAM (queued or being processed) at once
MAX_IN_FLIGHT = 2 * N_WORKERS

# Eye patches smaller than this (in pixels, either side) are treated as unusable
MIN_PATCH_SIDE = 3

# MediaPipe iris centers (478-landmark model)
LEFT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_CENTER_IDX = 473
# (outer corner, inner corner, upper lid, lower lid) for each eye
EAR_EYES = [(33, 133, 159, 145), (263, 362, 386, 374)]

METRICS = [
    "left_eye_brightness", "left_right_brightness_diff", "left_eye_brightness_sd",
    "interocular_distance_px", "interocular_distance_norm",
    "head_roll", "head_pitch", "head_yaw",
    "eye_aspect_ratio", "sharpness", "sensor_noise",
]
COLUMNS = ["video_id", "subject_id", "frame_idx", "dataset", "face_detected"] + METRICS
#######################

#%%
# Immerkaer (1996) "Fast Noise Variance Estimation" kernel
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)

def estimate_noise(gray):
    """
    Returns the Immerkaer estimate of the Gaussian noise sigma of a grayscale image.
    """
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float64), -1, NOISE_KERNEL)[1:-1, 1:-1]
    return math.sqrt(math.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2))

def laplace_sd(gray):
    """
    Returns the standard deviation of the Laplacian of a grayscale image.
    """
    return cv2.Laplacian(gray, cv2.CV_64F).std()

def eye_aspect_ratio(pts):
    """
    Returns the max EAR of the two eyes (lid gap / corner-to-corner width).
    """
    ears = []
    for outer, inner, top, bottom in EAR_EYES:
        width = np.linalg.norm(pts[outer] - pts[inner])
        if width > 0:
            ears.append(np.linalg.norm(pts[top] - pts[bottom]) / width)
    return max(ears) if ears else np.nan

def eye_patch(get_eye, gray, lm_arr):
    patch = get_eye(gray, lm_arr)
    if patch.shape[0] < MIN_PATCH_SIDE or patch.shape[1] < MIN_PATCH_SIDE:
        return None
    return patch

def frame_metrics(gray, landmarks):
    """
    Returns a dict of METRICS for one grayscale frame and its MediaPipe landmarks.
    """
    img_h, img_w = gray.shape
    m = dict.fromkeys(METRICS, np.nan)

    # getLeftEye/getRightEye take [[x, y, z], ...] normalized coordinates. Clip to the
    # frame so a partly out-of-frame eye doesn't turn into a negative (wrapping) slice.
    lm_arr = [[min(max(lm.x, 0.0), 1.0), min(max(lm.y, 0.0), 1.0), lm.z] for lm in landmarks]
    left = eye_patch(getLeftEye, gray, lm_arr)
    right = eye_patch(getRightEye, gray, lm_arr)

    if left is not None:
        m["left_eye_brightness"] = float(left.mean())
        m["left_eye_brightness_sd"] = float(left.std())
        m["sharpness"] = float(laplace_sd(left))
        m["sensor_noise"] = float(estimate_noise(left))
        if right is not None:
            m["left_right_brightness_diff"] = float(left.mean() - right.mean())

    pts = np.array([[lm.x * img_w, lm.y * img_h] for lm in landmarks])
    if len(pts) > RIGHT_IRIS_CENTER_IDX:
        iod = np.linalg.norm(pts[LEFT_IRIS_CENTER_IDX] - pts[RIGHT_IRIS_CENTER_IDX])
        m["interocular_distance_px"] = float(iod)
        m["interocular_distance_norm"] = float(iod / img_w)
    m["eye_aspect_ratio"] = float(eye_aspect_ratio(pts))

    pose = position_utils.calculate_head_pose(landmarks, img_w, img_h)
    if pose is not None:
        m["head_roll"], m["head_pitch"], m["head_yaw"] = (float(a) for a in pose)

    return m

#%%
def get_storage_client():
    # Service-account keys load directly; anything else (e.g. the user credentials
    # from `gcloud auth application-default login`) goes through google.auth.default()
    if GOOGLE_CREDENTIALS_JSON and json.loads(Path(GOOGLE_CREDENTIALS_JSON).read_text()).get("type") == "service_account":
        return storage.Client.from_service_account_json(GOOGLE_CREDENTIALS_JSON, project=GCP_PROJECT_ID)
    # Drop the ADC quota project: it gets billed for requests, and GCP_PROJECT_ID
    # has no billing account, which causes 403s on webcam-jspsych-frames
    creds, _ = google.auth.default()
    if hasattr(creds, "with_quota_project"):
        creds = creds.with_quota_project(None)
    return storage.Client(project=GCP_PROJECT_ID, credentials=creds)

def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading face_landmarker.task from Google...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

#%%
def video_fps(stream):
    fps = stream.guessed_rate or stream.average_rate
    fps = float(fps) if fps else 0.0
    # webm metadata is often missing/garbage; fall back to 30 fps if so
    if not np.isfinite(fps) or fps <= 0 or fps > 1000:
        fps = 30.0
    return fps

def process_video(args):
    """
    Decodes the video from memory and returns a DataFrame with one row per frame.
    """
    video_bytes, video_id, subject_id, dataset = args

    rows = []
    try:
        with av.open(io.BytesIO(video_bytes)) as container, \
                blink_utils.create_blink_landmarker(str(MODEL_PATH)) as landmarker:
            stream = container.streams.video[0]
            fps = video_fps(stream)

            # In VIDEO mode, MediaPipe requires strictly increasing timestamps
            last_timestamp_ms = -1
            for frame_idx, av_frame in enumerate(container.decode(stream)):
                rgb = av_frame.to_ndarray(format="rgb24")

                timestamp_ms = max(int((frame_idx / fps) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                face_detected = bool(result.face_landmarks)

                row = {
                    "video_id": video_id,
                    "subject_id": subject_id,
                    "frame_idx": frame_idx,
                    "dataset": dataset,
                    "face_detected": face_detected,
                }
                if face_detected:
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    row.update(frame_metrics(gray, result.face_landmarks[0]))
                rows.append(row)
    except (av.FFmpegError, IndexError) as e:
        # Corrupt/truncated webm or no video stream: keep any frames decoded before the error
        print(f"  WARNING: decode error in {video_id}: {e}")

    # Columns missing from a row (metrics on no-face frames) come out as NaN
    return pd.DataFrame(rows, columns=COLUMNS)

def append_rows(df, csv_path):
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

def load_processed_video_ids():
    done = set()
    for csv_path in [PUBLIC_CSV, PRIVATE_CSV]:
        if csv_path.exists():
            done.update(pd.read_csv(csv_path, usecols=["video_id"], dtype=str)["video_id"].unique())
    return done

#%%
def load_include_list(client):
    csv_bytes = client.bucket(INCLUDE_LIST_BUCKET).blob(INCLUDE_LIST_BLOB).download_as_bytes()
    include_df = pd.read_csv(io.BytesIO(csv_bytes), dtype={"subject_id": str})
    return set(include_df["subject_id"])

def iter_tar_videos(blob, processed_ids):
    """
    Streams the tar from the bucket and yields (video_id, video_bytes) for each
    unprocessed .webm, reading one member at a time into memory.
    """
    with blob.open("rb") as blob_reader, tarfile.open(fileobj=blob_reader, mode="r|*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".webm"):
                continue
            video_id = Path(member.name).stem
            if video_id in processed_ids:
                continue
            yield video_id, tar.extractfile(member).read()

def process_tar(pool, blob, dataset, public_subjects, processed_ids):
    n_new = 0

    def collect(done_futures):
        nonlocal n_new
        for future in done_futures:
            video_id, subject_id = pending.pop(future)
            df = future.result()
            if df.empty:
                print(f"  WARNING: no readable frames in {video_id}, skipping")
                continue
            append_rows(df, PUBLIC_CSV if subject_id in public_subjects else PRIVATE_CSV)
            processed_ids.add(video_id)
            n_new += 1

    pending = {}
    for video_id, video_bytes in iter_tar_videos(blob, processed_ids):
        subject_id = video_id.split("_")[0]
        future = pool.submit(process_video, (video_bytes, video_id, subject_id, dataset))
        pending[future] = (video_id, subject_id)
        if len(pending) >= MAX_IN_FLIGHT:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            collect(done)
    collect(wait(pending).done)

    print(f"  {n_new} new videos")

def sort_output_csvs():
    """
    Organize final CSVs by subject_id, then video_id, then frame_idx.
    """
    for csv_path in [PUBLIC_CSV, PRIVATE_CSV]:
        if csv_path.exists():
            df = pd.read_csv(csv_path, dtype={"video_id": str, "subject_id": str})
            df = df.drop_duplicates(subset=["video_id", "frame_idx"])
            df = df.sort_values(["subject_id", "video_id", "frame_idx"])
            df[COLUMNS].to_csv(csv_path, index=False)
            print(f"{csv_path.name}: {df['subject_id'].nunique()} subjects, "
                  f"{df['video_id'].nunique()} videos, {len(df)} frames")

#%%
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_model()

    client = get_storage_client()
    public_subjects = load_include_list(client)
    print(f"{len(public_subjects)} public subjects in include_list.csv")

    processed_ids = load_processed_video_ids()
    print(f"{len(processed_ids)} videos already processed")
    print(f"Using {N_WORKERS} workers")

    video_bucket = client.bucket(VIDEO_BUCKET)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        tar_blobs = [b for b in client.list_blobs(video_bucket, prefix=f"{DATASET}/") if b.name.endswith(".tar")]
        print(f"\n{DATASET}: {len(tar_blobs)} tar files")

        for i, blob in enumerate(tar_blobs):
            print(f"[{DATASET} {i+1}/{len(tar_blobs)}] {Path(blob.name).name}", flush=True)
            process_tar(pool, blob, DATASET, public_subjects, processed_ids)

    sort_output_csvs()

if __name__ == "__main__":
    main()
