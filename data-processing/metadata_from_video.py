# Extracts per-frame metadata from the webcam webm clips stored on Google Cloud.
#
# For every frame of every video, records:
#   video_id, subject_id, frame_idx, dataset,
#   face_detected, blink_score, laplace_variance, mean_brightness
#
# Subjects listed in include_list.csv are written to the public CSV; all
# others go to the private CSV. Output goes to <this dir>/video_metadata.
# The .tar files are streamed from the bucket and the videos are decoded in
# memory, so nothing but the CSVs is written to disk.
# The script is resumable: videos already present in the output CSVs are skipped.
#
# Runs on CHTC via extract_video_metadata.sub / extract_video_metadata.sh.

#%%
import io
import json
import os
import sys
import tarfile
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import av
import cv2
import google.auth
import numpy as np
import pandas as pd
from google.cloud import storage

#%%
current_dir = Path(__file__).parent.resolve()

sys.path.insert(0, str(current_dir))
import blink_utils

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
DATASETS = ["eyedata23", "eyedata24"]

MODEL_PATH = current_dir / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

OUTPUT_DIR = current_dir / "video_metadata"
PUBLIC_CSV = OUTPUT_DIR / "public_videos_metadata.csv"
PRIVATE_CSV = OUTPUT_DIR / "private_videos_metadata.csv"

# On CHTC, os.cpu_count() reports the whole machine, so the job passes its CPU request in
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
# Caps how many videos are held in RAM (queued or being processed) at once
MAX_IN_FLIGHT = 2 * N_WORKERS

COLUMNS = [
    "video_id", "subject_id", "frame_idx", "dataset",
    "face_detected", "blink_score", "laplace_variance", "mean_brightness",
]
#######################

#%%
def laplace_var(frame):
    """
    Returns laplacian variance of the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def mean_brightness(frame):
    """
    Return mean frame brightness.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

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
                frame = av_frame.to_ndarray(format="bgr24")

                timestamp_ms = max(int((frame_idx / fps) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                # blink_utils returns -1.0 for frames where MediaPipe finds no face
                b_score = blink_utils.frame_blink_score(landmarker, frame, timestamp_ms)
                face_detected = b_score >= 0
                rows.append({
                    "video_id": video_id,
                    "subject_id": subject_id,
                    "frame_idx": frame_idx,
                    "dataset": dataset,
                    "face_detected": face_detected,
                    "blink_score": float(b_score) if face_detected else np.nan,
                    "laplace_variance": float(laplace_var(frame)),
                    "mean_brightness": float(mean_brightness(frame)),
                })
    except (av.FFmpegError, IndexError) as e:
        # Corrupt/truncated webm or no video stream: keep any frames decoded before the error
        print(f"  WARNING: decode error in {video_id}: {e}")

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
        for dataset in DATASETS:
            tar_blobs = [b for b in client.list_blobs(video_bucket, prefix=f"{dataset}/") if b.name.endswith(".tar")]
            print(f"\n{dataset}: {len(tar_blobs)} tar files")

            for i, blob in enumerate(tar_blobs):
                print(f"[{dataset} {i+1}/{len(tar_blobs)}] {Path(blob.name).name}", flush=True)
                process_tar(pool, blob, dataset, public_subjects, processed_ids)

    sort_output_csvs()

if __name__ == "__main__":
    main()
