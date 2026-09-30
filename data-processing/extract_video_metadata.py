# Extracts per-frame metadata from the webcam webm clips stored on Google Cloud.
#
# For every frame of every video, records:
#   video_id, subject_id, frame_idx, dataset,
#   face_detected, blink_score, laplace_variance, mean_brightness
#
# Subjects listed in include_list.csv are written to the public CSV; all
# others go to the private CSV. Output goes to <project_root>/videos_metadata.
# The script is resumable: videos already present in the output CSVs are skipped.

#%%
import os
import sys
import tarfile
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import google.auth
import numpy as np
import pandas as pd
from google.cloud import storage

#%%
# This ensures paths are relative to the execution directory
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.parent  # Adjust as needed to reach project root
os.chdir(project_root)

sys.path.insert(0, str(current_dir))
import blink_utils

#%%
#### CONFIG ###########

# ---------------------------------------------------------------------------
# >>> GOOGLE ACCOUNT INFO: SUPPLY THIS <<<
# Authenticate with ONE of these options:
#   (a) Run once in a terminal:  gcloud auth application-default login
#       (log in with your Google account that has access to the buckets)
#       and leave GOOGLE_CREDENTIALS_JSON = None.
#   (b) Point GOOGLE_CREDENTIALS_JSON at a service-account key file (.json).
# GCP_PROJECT_ID: the Google Cloud project that owns the buckets
# (can be None if your default gcloud project is already set).
GOOGLE_CREDENTIALS_JSON = None   # e.g. "/home/you/keys/eyetracking-sa.json"
GCP_PROJECT_ID = "ardent-time-400616"
# ---------------------------------------------------------------------------

INCLUDE_LIST_BUCKET = "webcam-jspsych-frames"
INCLUDE_LIST_BLOB = "v1/include_list.csv"

VIDEO_BUCKET = "webcam-jspsych-webm"
DATASETS = ["eyedata23", "eyedata24"]

MODEL_PATH = project_root / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

OUTPUT_DIR = project_root / "videos_metadata"
PUBLIC_CSV = OUTPUT_DIR / "public_videos_metadata.csv"
PRIVATE_CSV = OUTPUT_DIR / "private_videos_metadata.csv"

# Downloaded .tar files are cached here so a crash doesn't force a re-download
DOWNLOAD_DIR = project_root / "webm_cache"
DELETE_TARS_AFTER_PROCESSING = False

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

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
    if GOOGLE_CREDENTIALS_JSON:
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
def process_video(args):
    """
    Returns a DataFrame with one row per frame of the video.
    """
    video_path, video_id, subject_id, dataset = args

    # blink_utils returns -1.0 for frames where MediaPipe finds no face
    blink_scores = blink_utils.extract_blendshape_blink_seq(str(video_path), model_path=str(MODEL_PATH))

    rows = []
    video = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        b_score = blink_scores[frame_idx] if frame_idx < len(blink_scores) else -1.0
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
        frame_idx += 1
    video.release()

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
    local_path = OUTPUT_DIR / "include_list.csv"
    client.bucket(INCLUDE_LIST_BUCKET).blob(INCLUDE_LIST_BLOB).download_to_filename(local_path)
    include_df = pd.read_csv(local_path, dtype={"subject_id": str})
    return set(include_df["subject_id"])

def process_tar(tar_path, dataset, public_subjects, processed_ids):
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp_dir, filter="data")

        jobs = []
        for video_path in sorted(Path(tmp_dir).rglob("*.webm")):
            video_id = video_path.stem
            if video_id in processed_ids:
                continue
            subject_id = video_id.split("_")[0]
            jobs.append((video_path, video_id, subject_id, dataset))

        if not jobs:
            return
        print(f"  {len(jobs)} new videos")

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for job, df in zip(jobs, pool.map(process_video, jobs)):
                video_id, subject_id = job[1], job[2]
                if df.empty:
                    print(f"  WARNING: no readable frames in {video_id}, skipping")
                    continue
                append_rows(df, PUBLIC_CSV if subject_id in public_subjects else PRIVATE_CSV)
                processed_ids.add(video_id)

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

    video_bucket = client.bucket(VIDEO_BUCKET)
    for dataset in DATASETS:
        dataset_dir = DOWNLOAD_DIR / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        tar_blobs = [b for b in client.list_blobs(video_bucket, prefix=f"{dataset}/") if b.name.endswith(".tar")]
        print(f"\n{dataset}: {len(tar_blobs)} tar files")

        for i, blob in enumerate(tar_blobs):
            tar_path = dataset_dir / Path(blob.name).name
            print(f"[{dataset} {i+1}/{len(tar_blobs)}] {tar_path.name}")
            if not tar_path.exists():
                # Download to a temp name so an interrupted download isn't mistaken for a complete one
                part_path = tar_path.with_suffix(".tar.part")
                blob.download_to_filename(part_path)
                part_path.rename(tar_path)

            process_tar(tar_path, dataset, public_subjects, processed_ids)

            if DELETE_TARS_AFTER_PROCESSING:
                tar_path.unlink()

    sort_output_csvs()

if __name__ == "__main__":
    main()
