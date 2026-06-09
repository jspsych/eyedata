import mediapipe as mp
import numpy as np
import cv2

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

MP_FACE_MESH = mp.solutions.face_mesh

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

def calculate_EAR(file_path):
    """
    Calculate max eye aspect ratio from face image.
    """

    # --- Main Processing Loop ---
    with MP_FACE_MESH.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

        image_height, image_width, _ = file_path.shape
        results = face_mesh.process(cv2.cvtColor(file_path, cv2.COLOR_BGR2RGB))

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

        video = cv2.VideoCapture(filename)

        EAR_list = [] 

        while video.isOpened():
            ret, frame = video.read()

            if ret:

                EAR = calculate_EAR(frame)
                EAR_list.append(EAR)
            
            else:
                break

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
    #   - Does this method assume normal distribution? Is this fine?
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
