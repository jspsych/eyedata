# Use position_utils to extract position data from JPGs
#%%
####
import tarfile
import os
from osfclient import OSF
from pathlib import Path
import mediapipe as mp
import cv2

import position_utils
from position_utils import LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES
####

#%%
# This ensures paths are relative to the execution directory
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.parent  # Adjust as needed to reach project root
os.chdir(project_root)

#%%
# First pull JPGs from OSF

#### CONFIG ###########
RETRIEVE_DATA = False

token_path = project_root / "osf-token.txt"
if token_path.exists():
    with open(token_path, "r") as f:
        osf_access_token = f.read().strip()
else:
    osf_access_token = ""  # Fallback if file doesn't exist

PRIVATE_JPG_OSF_COMPONENT = "xuyvw" 
PUBLIC_JPG_OSF_COMPONENT = "7s2ae"
#######################

#%%
# Check components have data
osf = OSF(token=osf_access_token)

components_to_check = {
    "PRIVATE_JPG": PRIVATE_JPG_OSF_COMPONENT,
    "PUBLIC_JPG": PUBLIC_JPG_OSF_COMPONENT,
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
private_dir = project_root / "private_data"
public_dir = project_root / "public_data"
data_dirs = [private_dir, public_dir]

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
    download_node_files(PRIVATE_JPG_OSF_COMPONENT, private_dir)
    download_node_files(PUBLIC_JPG_OSF_COMPONENT, public_dir)

    # Untar all files
    for data_dir in data_dirs:
        print(f"\n--- Extracting tarballs in {data_dir.parts[-1]} ---")
        json_tar = data_dir / "json.tar.gz"
        jpg_tar = data_dir / "jpg.tar.gz"

        if jpg_tar.exists():
            try: 
                print("Extracting JPG tarball...")
                with tarfile.open(jpg_tar, "r:gz") as tar:
                    tar.extractall(path=data_dir)
            except Exception as e:
                print(f"Error extracting JPG tarball: {e}")
                jpg_tar.unlink()  

#%%
# Extract position data from JPGs to parquet
print("\n--- Extracting Position Data from JPGs ---")

OVERWRITE = True 

def extract_position_data(jpg_dir, output_parquet):
    position_data = []
    for jpg_file in jpg_dir.glob("*.jpg"):
        try:
            # Actual extraction logic per-image

            image = cv2.imread(str(jpg_file))
            img_h, img_w, _ = image.shape

            mp_face_mesh = mp.solutions.face_mesh

            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # Relative iris distance (iris to image width ratio)
                    rel_dist = position_utils.calculate_iris_fraction(landmarks, LEFT_IRIS_INDICES, img_w)
                    angles = position_utils.calculate_head_pose(landmarks, img_w, img_h)
                    data = {
                        "filename": jpg_file.name,
                        "iris_to_image_width_ratio": rel_dist,
                        "head_pose_roll": angles[0],
                        "head_pose_pitch": angles[1],
                        "head_pose_yaw": angles[2],
                    }
            if data is not None:
                position_data.append(data)
        except Exception as e:
            print(f"Error processing {jpg_file.name}: {e}")
    
    if position_data:
        import pandas as pd
        df = pd.DataFrame(position_data)
        df.to_parquet(output_parquet, index=False)
        print(f"Position data saved to {output_parquet}.")
    else:
        print("No valid position data extracted.")

for data_dir in data_dirs:
    print(f"\nProcessing {data_dir.parts[-1]}...")
    jpg_dir = data_dir / "data/jpg"
    output_parquet = data_dir / "position_data.parquet"

    if output_parquet.exists() and not OVERWRITE:
        print(f"  Skipping {data_dir.parts[-1]} (position_data.parquet already exists).")
        continue

    if not jpg_dir.exists():
        print(f"  ERROR: JPG directory {jpg_dir} does not exist. Skipping.")
        continue

    try:
        extract_position_data(jpg_dir, output_parquet)
        print(f"  Successfully extracted position data to {output_parquet}.")
    except Exception as e:
        print(f"  ERROR: Failed to extract position data. Details: {e}")

#%%
# Upload parquets to OSF
# Helper function to upload back to OSF
def osf_upload_file(node_id, local_file_path):
    project = osf.project(node_id)
    storage = list(project.storages)[0]
    print(f"Uploading {local_file_path.name} to OSF...")
    with open(local_file_path, "rb") as f:
        storage.create_file(local_file_path.name, f, force=True)

print("\n--- Uploading Position Data to OSF ---")
for data_dir in data_dirs:
    output_parquet = data_dir / "position_data.parquet"
    if output_parquet.exists():
        node_id = PRIVATE_JPG_OSF_COMPONENT if "private" in data_dir.parts[-1] else PUBLIC_JPG_OSF_COMPONENT
        try:
            osf_upload_file(node_id, output_parquet)
            print(f"  Successfully uploaded {output_parquet.name} to OSF.")
        except Exception as e:
            print(f"  ERROR: Failed to upload {output_parquet.name}. Details: {e}")
    else:
        print(f"  WARNING: {output_parquet} does not exist. Skipping upload.")

# %%
# View parquet position data for verification
import pandas as pd
check_parquet = True
parquet_output_path = "public_data/position_data.parquet"
if check_parquet:
    position_df = pd.read_parquet(parquet_output_path)
    print(position_df.head())