import mediapipe as mp
import numpy as np
import cv2
from mediapipe.python.solutions import face_mesh as MP_FACE_MESH

def get_frames(file_path):
    """
    Converts video into an array of frames.
    """
    # Create a VideoCapture object 'video' to open the video file for reading.
    print("Program started. Press 'space' to pause or resume. Press 'Esc' to cancel.")
    video = cv2.VideoCapture(file_path)

    # Use a while loop to continuously read frames from the video until it's open.
    while video.isOpened():
        # Read the next video frame.
        ret, frame = video.read()
        
        # If 'ret' is True, the frame was read successfully.
        if ret:
            # Yield the current frame to the caller. 'yield' turns this function into a generator,
            # returning one frame at a time, but preserving the function state for the next call.
            yield frame
        else:
            # If 'ret' is False, it means there are no more frames to read in the video,
            # so we break the loop and stop the generator.
            break


    # Release the VideoCapture object to free up resources.
    video.release()

    # After the loop ends, yield 'None' to signal the end of the generator and stop iteration.
    # This helps to avoid raising StopIteration errors when the generator is exhausted.
    yield None

def get_frame(file_path, index):
    """
    Extract desired frame from video.
    """
    video = cv2.VideoCapture(file_path)
    
    # Directly set the position of the video reader to the specific index
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    
    ret, frame = video.read()
    video.release()
    
    if ret:
        return frame
    else:
        return None

# All points for the eye contours
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Corner points
LEFT_EYE_OUTER_CORNER_IDX = 33
LEFT_EYE_INNER_CORNER_IDX = 133
LEFT_TOP_IDX = 374
LEFT_BOTTOM_IDX = 386

RIGHT_EYE_OUTER_CORNER_IDX = 263
RIGHT_EYE_INNER_CORNER_IDX = 362
RIGHT_TOP_IDX = 145
RIGHT_BOTTOM_IDX = 159

def calculate_EAR(frame):
    """
    Calculate max eye aspect ratio from face image.
    """

    # --- Main Processing Loop ---
    with MP_FACE_MESH.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

        image_height, image_width, _ = frame.shape
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Helperto get all landmark points and convert to pixel coordinates
            get_point = lambda i: np.array([face_landmarks.landmark[i].x * image_width, face_landmarks.landmark[i].y * image_height])
            
            # Process each eye using the helper function
            p_left_outer, p_left_inner = get_point(LEFT_EYE_OUTER_CORNER_IDX), get_point(LEFT_EYE_INNER_CORNER_IDX)
            p_right_outer, p_right_inner = get_point(RIGHT_EYE_OUTER_CORNER_IDX), get_point(RIGHT_EYE_INNER_CORNER_IDX)
            
            left_top, left_bottom = get_point(LEFT_TOP_IDX), get_point(LEFT_BOTTOM_IDX)
            right_top, right_bottom = get_point(RIGHT_TOP_IDX), get_point(RIGHT_BOTTOM_IDX)

            # Norm of width vector divided by length vector.
            EAR_left = np.linalg.norm(np.array([left_top]) - np.array([left_bottom])) / np.linalg.norm(np.array([p_left_outer]) - np.array([p_left_inner]))
            EAR_right = np.linalg.norm(np.array([right_top]) - np.array([right_bottom])) / np.linalg.norm(np.array([p_right_outer]) - np.array([p_right_inner]))

            EAR_max = np.max([EAR_left, EAR_right])
        else:
            return False
    
    return EAR_max

def extract_EAR_seq(filename):
        """
        Calculate EAR for each frame, return list of all scores in order.
        """

        video = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)

        EAR_list = [] 

        # Initialize model once
        with MP_FACE_MESH.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
            while video.isOpened():
                ret, frame = video.read()

                if not ret:
                    break

                EAR = calculate_EAR(frame)
                EAR_list.append(EAR)

        video.release()


        return EAR_list

# TODO: Use permutation test, maybe with Monte Carlo sampling?
def get_blink_indices(ear_sequence, fps=30, z_threshold=3.5, window_size_s=1.0, global_ear_threshold=0.15):
    """
    Identifies blinks using Median and MAD for robust outlier detection.

    :return: Dictionary of blink indices (key) with eye aspect ratios (value).
    """
    ear_sequence = np.array(ear_sequence)
    window_num_frames = int(fps * window_size_s)
    shift = window_num_frames // 2

    blink_dict = {}
    
    # Constant to convert MAD to an estimate of Standard Deviation
    # (for normal distributions, sigma approx 1.4826 * MAD) 
    MAD_SCALE_FACTOR = 1.4826

    for i, ear in enumerate(ear_sequence):
        # Define window bounds
        start = max(0, i - shift)
        end = min(len(ear_sequence), i + shift + 1) # +1 because slice end is exclusive
        
        window = ear_sequence[start:end]

        # Calculate Median (Robust Central Tendency)
        window_median = np.median(window)

        # Calculate MAD (Robust Variation)
        # Median of the absolute deviations from the data's median
        mad = np.median(np.abs(window - window_median))
        
        # Estimate "Sigma" (Standard Deviation) from MAD
        sigma_estimate = mad * MAD_SCALE_FACTOR
        
        # Define Threshold
        # We look for values significantly BELOW the median
        local_ear_threshold = window_median - (z_threshold * sigma_estimate)

        # Determine if blink
        # Only use the local threshold if less than 0.18. I noticed lots of false positives at higher value.
        if (ear < 0.18 and ear < local_ear_threshold) or ear < global_ear_threshold:
            blink_dict[i] = ear

    return blink_dict

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_blendshape_blink_seq(video_path, model_path="face_landmarker.task"):
    """
    Processes video and extracts blink scores using MediaPipe Blendshapes.
    Returns a list of scores (0.0 to 1.0) representing eye closure per frame.
    """
    # 1. Configure the Face Landmarker Options
    base_options = python.BaseOptions(model_asset_path=model_path)
    
    # We use VIDEO mode to leverage temporal tracking across frames
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    
    blink_scores_list = []

    # 2. Initialize the model ONCE using context management
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # In VIDEO mode, MediaPipe requires timestamps
        frame_index = 0 
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            # Convert OpenCV BGR format to MediaPipe RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calculate timestamp in milliseconds
            timestamp_ms = int((frame_index / fps) * 1000)
            
            # 3. Detect features and extract blendshapes
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.face_blendshapes:
                # Get blendshapes for the primary face
                blendshapes = detection_result.face_blendshapes[0]
                
                left_blink_score = 0.0
                right_blink_score = 0.0
                
                # Iterate through the 52 blendshapes to find the blink classifications
                for category in blendshapes:
                    if category.category_name == 'eyeBlinkLeft':
                        left_blink_score = category.score
                    elif category.category_name == 'eyeBlinkRight':
                        right_blink_score = category.score
                        
                # Take the max score to account for winks or uneven closures
                max_blink_score = max(left_blink_score, right_blink_score)
                blink_scores_list.append(max_blink_score)
                
            else:
                # If no face is found, default to 0.0
                blink_scores_list.append(0.0)
                
            frame_index += 1
            
        video.release()
        
    return blink_scores_list

def get_blendshape_blink_indices(blink_scores_list, threshold=0.6):
    """
    Because blendshapes output a normalized score from 0.0 to 1.0, 
    we can use a static threshold. No MAD or sliding window needed!
    """
    blink_dict = {}
    
    for i, score in enumerate(blink_scores_list):
        if score > threshold:
            blink_dict[i] = score
            
    return blink_dict
    

if __name__ == "__main__":
    # Test the blink detection on a sample video
    import os
    import matplotlib.pyplot as plt
    from pathlib import Path
    import urllib.request

    

    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent.parent.parent  # Adjust as needed to reach project root
    os.chdir(project_root)

    model_path = "face_landmarker.task"

    # Check if the file already exists
    if not os.path.exists(model_path):
        print("Downloading face_landmarker.task from Google...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete!")
    else:
        print("Model already exists.")

    test_video_path = "test_videos/video_1.avi"  # Replace with your test video path

    # Get blinks using custom EAR functions:

    # blink_score_list = extract_EAR_seq(test_video_path)
    # blink_indices = get_blink_indices(ear_sequence, fps=30, z_threshold=3.5, window_size_s=1.0, global_ear_threshold=0.15)

    # blink_frames = list(blink_indices.keys())
    # blink_values = list(blink_indices.values())
    # print(f"Detected blinks at frame indices: {blink_frames}")

    # Get blinks using MediaPipe Blendshapes:
    blink_scores_list = extract_blendshape_blink_seq(test_video_path, model_path=model_path)

    thres_val = 0.6
    blink_indices = get_blendshape_blink_indices(blink_scores_list, threshold=thres_val)

    blink_frames = list(blink_indices.keys())
    print(f"Detected blinks at frame indices: {blink_frames}")

    blink_values = list(blink_indices.values())

    # ==========================================
    # Plotting the EAR Sequence and Blink Indices
    # ==========================================
    # Video 1 blink frames: [10,11, 40, 41, 42, 80, 81, 82, 100, 101, 103, 131, 132, 133, 163, 164, 191, 192, 193, 229, 230, 259, 260]

    plt.figure(figsize=(14, 6))

    # 1. Plot the continuous EAR signal
    plt.plot(blink_scores_list, label="Blink Scores", color="#1f77b4", linewidth=1.5)

    # 2. Scatter plot the detected blinks on top of the signal
    plt.scatter(blink_frames, blink_values, color="red", s=60, zorder=5, label="Predicted Blinks")

    # 3. Draw a horizontal line representing your global hard-threshold
    plt.axhline(y=thres_val, color='gray', linestyle='--', alpha=0.7, label=f"Global Threshold ({thres_val})")

    # Formatting the chart
    plt.title("Blink Scores over Time", fontsize=16)
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Blink Score", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Display the plot
    plt.show()