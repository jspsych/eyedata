# 5/19/26 converted process-subject-metadata.R to Python mainly using Google Gemini

#%%
import os
import re
import tarfile
from pathlib import Path
import pandas as pd
from osfclient.api import OSF

#%%
# This ensures paths are relative to the execution directory
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.parent  # Adjust as needed to reach project root
os.chdir(project_root)

#%%
#### CONFIG ###########
RETRIEVE_DATA = False 
UPLOAD_TO_OSF = True 

token_path = project_root / "osf-token.txt"
if token_path.exists():
    with open(token_path, "r") as f:
        osf_access_token = f.read().strip()
else:
    osf_access_token = ""  # Fallback if file doesn't exist

PRIVATE_JSON_OSF_COMPONENT = "vmdu6"
PRIVATE_JPG_OSF_COMPONENT = "xuyvw" 
PRIVATE_WEBM_OSF_COMPONENT = "34t8d" 
PRIVATE_OSF_OVERALL_COMPONENT = "2wcgp"

PUBLIC_JPG_OSF_COMPONENT = "7s2ae"
PUBLIC_JSON_OSF_COMPONENT = "6cn7a"
PUBLIC_WEBM_OSF_COMPONENT = "Urm78" 

private_dir = project_root / "private_data"
public_dir = project_root / "public_data"
data_dirs = [private_dir, public_dir]
#######################

#%%
# Check components have data
osf = OSF(token=osf_access_token)

components_to_check = {
    "PRIVATE_JSON": PRIVATE_JSON_OSF_COMPONENT,
    "PRIVATE_JPG": PRIVATE_JPG_OSF_COMPONENT,
    "PRIVATE_OSF_OVERALL": PRIVATE_OSF_OVERALL_COMPONENT,
    "PUBLIC_JPG": PUBLIC_JPG_OSF_COMPONENT,
    "PUBLIC_JSON": PUBLIC_JSON_OSF_COMPONENT,
}

print("--- Scanning OSF Components ---")

for label, node_id in components_to_check.items():
    print(f"\nComponent: {label} ({node_id})")
    try:
        project = osf.project(node_id)
        file_count = 0
        for storage in project.storages:
            for osf_file in storage.files:
                file_count += 1
                if file_count <= 5:
                    print(f"  - Found file: {osf_file.name} ({osf_file.size} bytes)")
                elif file_count == 6:
                    print("  - ... and more files exist.")
        
        if file_count == 0:
            print("  WARNING: This component is EMPTY (0 files found).")
        else:
            print(f"  Total files in component: {file_count}")
            
    except Exception as e:
        print(f"  ERROR: Could not access component. Details: {e}")

#%%

if RETRIEVE_DATA:
    private_dir.mkdir(exist_ok=True)
    public_dir.mkdir(exist_ok=True)

    def download_node_files(node_id, target_dir, file_filter=None):
        project = osf.project(node_id)
        for storage in project.storages:
            for osf_file in storage.files:
                if file_filter is None or file_filter(osf_file.name):
                    local_path = target_dir / osf_file.name

                    if local_path.exists():
                        print(f"Skipping {osf_file.name} (already downloaded).")
                        continue

                    print(f"Downloading {osf_file.name}...")
                    with open(local_path, "wb") as f:
                        osf_file.write_to(f)

    print("\n--- Downloading Data ---")
    download_node_files(PRIVATE_JSON_OSF_COMPONENT, private_dir)
    download_node_files(PRIVATE_JPG_OSF_COMPONENT, private_dir)
    download_node_files(PUBLIC_JSON_OSF_COMPONENT, public_dir)
    download_node_files(PUBLIC_JPG_OSF_COMPONENT, public_dir)

    # Untar all files
    for data_dir in data_dirs:
        print(f"\n--- Extracting tarballs in {data_dir.parts[-1]} ---")
        json_tar = data_dir / "json.tar.gz"
        jpg_tar = data_dir / "jpg.tar.gz"

        if json_tar.exists():
            print("Extracting JSON tarball...")
            target_dir = data_dir / "json"
            target_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(json_tar, "r:gz") as tar:
                for member in tar.getmembers():
                    member_path = Path(member.name)
                    if not member_path.parts:
                        continue
                    if member_path.parts[0] == "json":
                        relative_path = Path(*member_path.parts[1:])
                    else:
                        relative_path = member_path
                    if str(relative_path) == "." or relative_path.name == "":
                        continue

                    final_path = target_dir / relative_path
                    final_path.parent.mkdir(parents=True, exist_ok=True)

                    with tar.extractfile(member) as source_file:
                        if source_file:
                            final_path.write_bytes(source_file.read())

        if jpg_tar.exists():
            try: 
                print("Extracting JPG tarball...")
                with tarfile.open(jpg_tar, "r:gz") as tar:
                    tar.extractall(path=data_dir)
            except Exception as e:
                print(f"Error extracting JPG tarball: {e}")
                jpg_tar.unlink()  


#%%
# Helper function to upload back to OSF
def osf_upload_file(node_id, local_file_path):
    project = osf.project(node_id)
    storage = list(project.storages)[0]
    print(f"Uploading {local_file_path.name} to OSF...")
    with open(local_file_path, "rb") as f:
        storage.create_file(local_file_path.name, f, force=True)

# --- Data Processing Loop ---
for data_dir in data_dirs:
    print(f"\n--- Processing data in {data_dir.parts[-1]} ---")
    json_path = data_dir / "json"
    
    json_files = [f for f in json_path.glob("*.json") if re.match(r"^\w{8}\.json$", f.name)]

    # Read base JSON files
    json_data_list = [pd.read_json(f) for f in json_files]
    json_data = pd.concat(json_data_list, ignore_index=True) if json_data_list else pd.DataFrame()

    if json_data.empty:
        print(f"No JSON data found in {data_dir.parts[-1]}. Skipping.")
        continue

    # Prolific info setup
    prolific_info = json_data[["subject_id", "prolific_id"]].drop_duplicates(subset=["subject_id"], keep="first")

    # Filter experimental data
    data_trials = ["webcam-type", "glasses", "browser-check", "public-videos", "research-videos"]
    experiment_data = json_data[json_data["task"].isin(data_trials) | (json_data["trial_type"] == "html-video-response")]

    # Find subjects who gave permission
    people_who_give_permission_for_videos = experiment_data[
        (experiment_data["task"] == "public-videos") & (experiment_data["response"] == 0)
    ]["subject_id"].unique()

    # Process JPG file counts
    jpg_path = data_dir / "data/jpg"
    if not jpg_path.exists():
        jpg_path = data_dir / "jpg" # Fallback
        
    jpg_files = [f.name for f in jpg_path.glob("*.jpg")] if jpg_path.exists() else []
    print(f"{len(jpg_files)} JPG files found in {jpg_path}.")

    if jpg_files:
        jpg_df = pd.DataFrame({"file_name": jpg_files})
        jpg_df["subject_id"] = jpg_df["file_name"].str.slice(0, 8)
        counts = jpg_df.groupby("subject_id").size().reset_index(name="n")
        subjects_with_complete_data = counts[counts["n"] == 144]["subject_id"].tolist()
    else:
        subjects_with_complete_data = []

    print(f"Subjects with complete data (144 JPGs): {len(subjects_with_complete_data)}")

    # Allowed subjects list
    allowed_subjects_list = list(set(subjects_with_complete_data).intersection(set(people_who_give_permission_for_videos)))
    print(f"Subjects with complete data and permissions: {len(allowed_subjects_list)}")

    # --- Feature Extraction ---
    
    # Webcam info
    webcam_info = experiment_data[experiment_data["task"] == "webcam-type"][["subject_id", "response"]].copy()
    webcam_info["webcam_type"] = webcam_info["response"].map({0: "integrated", 1: "external"})
    webcam_info = webcam_info.drop(columns=["response"])

    # Glasses info
    glasses_info = experiment_data[experiment_data["task"] == "glasses"][["subject_id", "response"]].copy()
    glasses_info["wore_glasses"] = glasses_info["response"].map({0: "no", 1: "yes"})
    glasses_info = glasses_info.drop(columns=["response"])

    # Screen dimensions
    screen_info = experiment_data[experiment_data["task"] == "browser-check"][["subject_id", "aspect_ratio", "width", "height"]]

    # Dot detection task
    dot_detection = json_data[json_data["task"] == "dot_detection"]
    if not dot_detection.empty:
        detection_task_data = (
            dot_detection.groupby("subject_id")["rt"]
            .agg(
                mean="mean", median="median", sd="std",
                prop_200ms=lambda x: (x < 200).sum() / len(x),
                prop_400ms=lambda x: (x < 400).sum() / len(x),
                prop_800ms=lambda x: (x < 800).sum() / len(x),
                prop_1600ms=lambda x: (x < 1600).sum() / len(x),
                prop_3200ms=lambda x: (x < 3200).sum() / len(x),
            ).reset_index()
        )
        detection_task_data.rename(
            columns={
                "prop_200ms": "prop<200ms", "prop_400ms": "prop<400ms",
                "prop_800ms": "prop<800ms", "prop_1600ms": "prop<1600ms",
                "prop_3200ms": "prop<3200ms",
            }, inplace=True
        )

    # Interaction Data
    interaction_files = list(json_path.glob("*_interactions.json"))
    interaction_data_list = []
    for x in interaction_files:
        df = pd.read_json(x)
        df["subject_id"] = x.name[:8]
        interaction_data_list.append(df)

    interaction_data = pd.concat(interaction_data_list, ignore_index=True) if interaction_data_list else pd.DataFrame()

    if not interaction_data.empty:
        interaction_data_wide = (
            interaction_data[(interaction_data["trial"] >= 9) & (interaction_data["trial"] <= 340)]
            .groupby(["subject_id", "event"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        fullscreenexit_trial = (
            interaction_data[interaction_data["event"] == "fullscreenexit"]
            .sort_values("trial")
            .groupby("subject_id")
            .first()
            .reset_index()[["subject_id", "trial"]]
            .rename(columns={"trial": "fullscreen_exit_trial"})
        )
    
    # Position features
    position_parquet = data_dir / "position_data.parquet"

    if position_parquet.exists():
        all_position_data = pd.read_parquet(position_parquet)
        all_position_data = all_position_data[["filename", "iris_to_image_width_ratio", "head_pose_roll", "head_pose_pitch", "head_pose_yaw"]]
        all_position_data["subject_id"] = all_position_data["filename"].str.slice(0, 8)

        numeric_cols = [
            "iris_to_image_width_ratio",
            "head_pose_roll",
            "head_pose_pitch",
            "head_pose_yaw"
        ]

        # Group by subject id and apply aggregation functions
        position_data = all_position_data.groupby("subject_id")[numeric_cols].agg(['mean', 'median', 'std']).reset_index()

        # Flatten MultiIndex columns
        position_data.columns = ['subject_id'] + [f"{col}_{stat}" for col in numeric_cols for stat in ['mean', 'median', 'std']]
    else:
        print(f"Position data parquet not found in {data_dir.parts[-1]}. Skipping position features.")


    # --- Merge Everything ---
    metadata_df = pd.DataFrame({"subject_id": allowed_subjects_list})
    metadata_df = metadata_df.merge(prolific_info, on="subject_id", how="left")
    metadata_df = metadata_df.merge(webcam_info, on="subject_id", how="left")
    metadata_df = metadata_df.merge(glasses_info, on="subject_id", how="left")
    metadata_df = metadata_df.merge(screen_info, on="subject_id", how="left")
    metadata_df = metadata_df.merge(position_data, on="subject_id", how="left") if position_parquet.exists() else metadata_df

    if not dot_detection.empty:
        metadata_df = metadata_df.merge(detection_task_data, on="subject_id", how="left")

    if not interaction_data.empty:
        metadata_df = metadata_df.merge(interaction_data_wide, on="subject_id", how="left")
        metadata_df = metadata_df.merge(fullscreenexit_trial, on="subject_id", how="left")

    metadata_df = metadata_df.drop_duplicates(subset=["subject_id"])

    # --- Save to Parquet ---
    parquet_output_path = data_dir / "allowed_subject_metadata.parquet"
    metadata_df.to_parquet(parquet_output_path, index=False, engine="fastparquet")
    print(f"Saved Parquet metadata to {parquet_output_path.name}")

    if UPLOAD_TO_OSF:
        osf_upload_file(PRIVATE_OSF_OVERALL_COMPONENT, parquet_output_path)

    # --- Public JPG Tarball Generation ---
    public_jpg_files = [
        f for f in jpg_files 
        if re.match(r"^[^_]+", f) and re.match(r"^[^_]+", f).group(0) in allowed_subjects_list
    ]

    public_tar_path = data_dir / "public-jpg.tar.gz"
    
    if public_jpg_files:
        print(f"Creating public tarball with {len(public_jpg_files)} allowed images...")
        with tarfile.open(public_tar_path, "w:gz") as tar:
            for f in public_jpg_files:
                file_to_add = jpg_path / f
                if file_to_add.exists():
                    tar.add(file_to_add, arcname=f"data/jpg/{f}")

        if UPLOAD_TO_OSF:
            osf_upload_file(PUBLIC_JPG_OSF_COMPONENT, public_tar_path)
    else:
        print("No valid public JPGs found to archive.")

print("\n--- Pipeline Complete! ---")

# %%
# View parquet metadata for verification
check_parquet = True
parquet_output_path = "public_data/allowed_subject_metadata.parquet"
if check_parquet:
    metadata_df = pd.read_parquet(parquet_output_path)
    print(metadata_df.head())
