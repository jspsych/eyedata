# 6/19/26 converted process-subject-metadata.R to Python mainly using Google Gemini

#%%
import json
import os
import re
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from osfclient.api import OSF
#%%


#%%
# This ensures paths are relative to the execution directory
# base_dir = Path(".").resolve()
base_dir = Path(os.getcwd()).resolve()
#%%

#%%
#### CONFIG ###########
RETRIEVE_DATA = True

token_path = base_dir / "osf-token.txt"
if token_path.exists():
    with open(token_path, "r") as f:
        osf_access_token = f.read().strip()
else:
    osf_access_token = ""  # Fallback if file doesn't exist

PRIVATE_JSON_OSF_COMPONENT = "vmdu6"
PRIVATE_JPG_OSF_COMPONENT = "p3xyj"
PRIVATE_WEBM_OSF_COMPONENT = "34t8d" # Currently empty
PRIVATE_OSF_OVERALL_COMPONENT = "2wcgp"

PUBLIC_JPG_OSF_COMPONENT = "7s2ae"
PUBLIC_JSON_OSF_COMPONENT = "6cn7a"
PUBLIC_WEBM_OSF_COMPONENT = "Urm78" # Currently empty
#######################
#%%

#%%
# Check components have data
osf = OSF(token=osf_access_token)

# Put your component configuration into a dictionary for easy looping
components_to_check = {
    "PRIVATE_JSON": PRIVATE_JSON_OSF_COMPONENT,
    "PRIVATE_JPG": PRIVATE_JPG_OSF_COMPONENT,
    "PRIVATE_WEBM": PRIVATE_WEBM_OSF_COMPONENT,
    "PRIVATE_OSF_OVERALL": PRIVATE_OSF_OVERALL_COMPONENT,
    "PUBLIC_JPG": PUBLIC_JPG_OSF_COMPONENT,
    "PUBLIC_JSON": PUBLIC_JSON_OSF_COMPONENT,
    "PUBLIC_WEBM": PUBLIC_WEBM_OSF_COMPONENT,
}

print("--- Scanning OSF Components ---")

for label, node_id in components_to_check.items():
    print(f"\nComponent: {label} ({node_id})")
    try:
        project = osf.project(node_id)
        file_count = 0
        
        # Components can have multiple storage providers, though 'osfstorage' is default
        for storage in project.storages:
            for osf_file in storage.files:
                file_count += 1
                # Print the first few files as a sample, or remove the if-statement to see all
                if file_count <= 5:
                    print(f"  - Found file: {osf_file.name} ({osf_file.size} bytes)")
                elif file_count == 6:
                    print("  - ... and more files exist.")
        
        if file_count == 0:
            print("  WARNING: This component is EMPTY (0 files found).")
        else:
            print(f"  Total files in component: {file_count}")
            
    except Exception as e:
        print(f"  ERROR: Could not access component. Check your permissions/token. Details: {e}")
#%%

#%%
data_dir = base_dir / "data"

if RETRIEVE_DATA:
    data_dir.mkdir(exist_ok=True)

    # Authenticate with OSF
    osf = OSF(token=osf_access_token)

    # Helper function to download files from an OSF node
    def download_node_files(node_id, target_dir, file_filter=None):
        project = osf.project(node_id)
        for storage in project.storages:
            for osf_file in storage.files:
                if file_filter is None or file_filter(osf_file.name):
                    local_path = target_dir / osf_file.name
                    print(f"Downloading {osf_file.name}...")
                    with open(local_path, "wb") as f:
                        osf_file.write_to(f)

    # Download private JSON files
    download_node_files(PRIVATE_JSON_OSF_COMPONENT, data_dir)

    # Download specific private JPG part file
    download_node_files(
        PRIVATE_JPG_OSF_COMPONENT,
        data_dir,
        file_filter=lambda name: name == "jpg.tar.gz.part.aa",
    )

    # Rename file
    part_file = data_dir / "jpg.tar.gz.part.aa"
    jpg_tar = data_dir / "jpg.tar.gz"
    if part_file.exists():
        part_file.rename(jpg_tar)

    # Untar files
    json_tar = data_dir / "json.tar.gz"
    if json_tar.exists():
        with tarfile.open(json_tar, "r:gz") as tar:
            tar.extractall(path=data_dir / "json")

    if jpg_tar.exists():
        with tarfile.open(jpg_tar, "r:gz") as tar:
            tar.extractall(path=data_dir / "jpg")

#%%

# --- Data Processing ---

#%%
json_path = data_dir / "json"
# Matches exactly 8 word characters followed by .json (e.g., abcdefgh.json)
json_files = [
    f for f in json_path.glob("*.json") if re.match(r"^\w{8}\.json$", f.name)
]

# Read all JSON files and bind rows
json_data_list = [pd.read_json(f) for f in json_files]
json_data = (
    pd.concat(json_data_list, ignore_index=True)
    if json_data_list
    else pd.DataFrame()
)

# Prolific info setup
prolific_info = (
    json_data[["subject_id", "prolific_id"]].drop_duplicates(
        subset=["subject_id"], keep="first"
    )
    if not json_data.empty
    else pd.DataFrame()
)

data_trials = [
    "webcam-type",
    "glasses",
    "browser-check",
    "public-videos",
    "research-videos",
]

# Filter experimental data
experiment_data = json_data[
    json_data["task"].isin(data_trials)
    | (json_data["trial_type"] == "html-video-response")
]

# Find subjects who gave permission
people_who_give_permission_for_videos = experiment_data[
    (experiment_data["task"] == "public-videos")
    & (experiment_data["response"] == 0)
]["subject_id"].unique()

# Process JPG file counts
jpg_path = data_dir / "jpg"
jpg_files = [f.name for f in jpg_path.glob("*.jpg")]

jpg_df = pd.DataFrame({"file_name": jpg_files})
jpg_df["subject_id"] = jpg_df["file_name"].str.slice(0, 8)

counts = jpg_df.groupby("subject_id").size().reset_index(name="n")
subjects_with_complete_data = counts[counts["n"] == 144]["subject_id"].tolist()

# Find intersection of complete data and permission
subjects_with_complete_data_and_permission = list(
    set(subjects_with_complete_data).intersection(
        set(people_who_give_permission_for_videos)
    )
)

# Save allowed subjects to JSON
output_permissions_json = data_dir / "subjects-with-permission.json"
with open(output_permissions_json, "w") as f:
    json.dump(subjects_with_complete_data_and_permission, f)


# Helper function to upload back to OSF
def osf_upload_file(node_id, local_file_path):
    project = osf.project(node_id)
    # Target the primary osfstorage provider
    storage = list(project.storages)[0]
    print(f"Uploading {local_file_path.name} to OSF...")
    with open(local_file_path, "rb") as f:
        storage.create_file(local_file_path.name, f, force=True)


if RETRIEVE_DATA:
    osf_upload_file(PRIVATE_OSF_OVERALL_COMPONENT, output_permissions_json)

# Webcam info processing
webcam_info = experiment_data[experiment_data["task"] == "webcam-type"][
    ["subject_id", "response"]
].copy()
webcam_info["webcam_type"] = webcam_info["response"].map(
    {0: "integrated", 1: "external"}
)
webcam_info = webcam_info.drop(columns=["response"])

# Glasses info processing
glasses_info = experiment_data[experiment_data["task"] == "glasses"][
    ["subject_id", "response"]
].copy()
glasses_info["wore_glasses"] = glasses_info["response"].map(
    {0: "no", 1: "yes"}
)
glasses_info = glasses_info.drop(columns=["response"])

# Screen dimensions processing
screen_info = experiment_data[experiment_data["task"] == "browser-check"][
    ["subject_id", "aspect_ratio", "width", "height"]
]

# Dot detection task analytics
dot_detection = json_data[json_data["task"] == "dot_detection"]
if not dot_detection.empty:
    detection_task_data = (
        dot_detection.groupby("subject_id")["rt"]
        .agg(
            mean="mean",
            median="median",
            sd="std",
            prop_200ms=lambda x: (x < 200).sum() / len(x),
            prop_400ms=lambda x: (x < 400).sum() / len(x),
            prop_800ms=lambda x: (x < 800).sum() / len(x),
            prop_1600ms=lambda x: (x < 1600).sum() / len(x),
            prop_3200ms=lambda x: (x < 3200).sum() / len(x),
        )
        .reset_index()
    )

    # Rename columns to match exact R naming conventions
    detection_task_data.rename(
        columns={
            "prop_200ms": "prop<200ms",
            "prop_400ms": "prop<400ms",
            "prop_800ms": "prop<800ms",
            "prop_1600ms": "prop<1600ms",
            "prop_3200ms": "prop<3200ms",
        },
        inplace=True,
    )

# Interaction Data aggregation
interaction_files = list(json_path.glob("*_interactions.json"))
interaction_data_list = []

for x in interaction_files:
    df = pd.read_json(x)
    # Extracts the first 8 characters from the file name safely
    df["subject_id"] = x.name[:8]
    interaction_data_list.append(df)

interaction_data = (
    pd.concat(interaction_data_list, ignore_index=True)
    if interaction_data_list
    else pd.DataFrame()
)

# Reshape Interaction Data (Pivot Wider)
if not interaction_data.empty:
    interaction_data_wide = (
        interaction_data[
            (interaction_data["trial"] >= 9) & (interaction_data["trial"] <= 340)
        ]
        .groupby(["subject_id", "event"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Track first instance of fullscreen exit
    fullscreenexit_trial = (
        interaction_data[interaction_data["event"] == "fullscreenexit"]
        .sort_values("trial")
        .groupby("subject_id")
        .first()
        .reset_index()[["subject_id", "trial"]]
        .rename(columns={"trial": "fullscreen_exit_trial"})
    )

# Filter and create public images archive
public_jpg_files = []
for f in jpg_files:
    match = re.match(r"^[^_]+", f)
    if (
        match
        and match.group(0) in subjects_with_complete_data_and_permission
    ):
        public_jpg_files.append(f)

public_tar_path = data_dir / "public-jpg.tar.gz"

# Generate gzipped tarball preserving the directory structure
with tarfile.open(public_tar_path, "w:gz") as tar:
    for f in public_jpg_files:
        file_to_add = jpg_path / f
        if file_to_add.exists():
            # arcname structures the paths inside the archive to look like 'data/jpg/filename.jpg'
            tar.add(file_to_add, arcname=f"data/jpg/{f}")

if RETRIEVE_DATA:
    osf_upload_file(PUBLIC_JPG_OSF_COMPONENT, public_tar_path)

#%%
