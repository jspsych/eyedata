#%%
from pathlib import Path

import mediapipe as mp
import numpy as np
import math
import cv2
# from scipy.spatial.transform import Rotation as R

# Iris landmark indices based on MediaPipe's face mesh model
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

def calculate_iris_fraction(landmarks, eye_indices, image_width):
    # Get the x-coordinates of the leftmost and rightmost iris landmarks
    x_coords = [landmarks[idx].x * image_width for idx in eye_indices]
    iris_pixel_width = max(x_coords) - min(x_coords)
    return iris_pixel_width / image_width


# def calculate_head_pose(landmarks, img_w, img_h):
#     # Standardized 3D Model Points (Aligned to Image Space: +X is Right, +Y is Down)
#     model_points = np.array([
#         (0.0, 0.0, 0.0),             # Nose tip
#         (0.0, 330.0, -65.0),         # Chin (+Y is down)
#         (225.0, -170.0, -135.0),     # Left Eye, Left Corner (User's L / Image Right -> +X is right, -Y is up)
#         (-225.0, -170.0, -135.0),    # Right Eye, Right Corner (User's R / Image Left -> -X is left, -Y is up)
#         (150.0, 150.0, -125.0),      # Left Mouth Corner (User's L / Image Right -> +X is right, +Y is down)
#         (-150.0, 150.0, -125.0)      # Right Mouth Corner (User's R / Image Left -> -X is left, +Y is down)
#     ])

#     # MediaPipe: 263 is User's Left Eye Outer. 33 is User's Right Eye Outer.
#     # MediaPipe: 287 is User's Left Mouth. 57 is User's Right Mouth.
#     image_points = np.array([
#         (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
#         (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
#         (landmarks[263].x * img_w, landmarks[263].y * img_h), # Left eye left corner
#         (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Right eye right corner
#         (landmarks[287].x * img_w, landmarks[287].y * img_h), # Left mouth corner
#         (landmarks[57].x * img_w, landmarks[57].y * img_h)    # Right mouth corner
#     ], dtype="double")

#     # Camera Internals Matrix
#     focal_length = img_w
#     center = (img_w / 2, img_h / 2)
#     camera_matrix = np.array(
#         [[focal_length, 0, center[0]],
#          [0, focal_length, center[1]],
#          [0, 0, 1]], dtype="double"
#     )

#     dist_coeffs = np.zeros((4, 1))

#     success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
#     if not success:
#         return None, None, (0, 0, 0)

#     # Robustly decompose rotation matrix using OpenCV's built-in RQ decomposition 
#     rmat, _ = cv2.Rodrigues(rvec)
#     angles = R.from_matrix(rmat).as_euler('xyz', degrees=True)
    
#     pitch = angles[0]
#     yaw = angles[1]
#     roll = angles[2]

#     return rvec, tvec, (roll, pitch, yaw)

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0

    return np.degrees([pitch, yaw, roll])


def calculate_head_pose(landmarks, img_w, img_h):
    # Your original 3D Model Points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, 330.0, -65.0),         # Chin
        (225.0, -170.0, -135.0),     # Left Eye
        (-225.0, -170.0, -135.0),    # Right Eye
        (150.0, 150.0, -125.0),      # Left Mouth
        (-150.0, 150.0, -125.0)      # Right Mouth
    ], dtype="double")

    # Your original 2D mappings
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[152].x * img_w, landmarks[152].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[287].x * img_w, landmarks[287].y * img_h),
        (landmarks[57].x * img_w, landmarks[57].y * img_h)
    ], dtype="double")

    # Camera Internals
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    # Using SOLVEPNP_EPNP as it does not require an initial guess and is very stable for individual frames
    success, rvec, tvec = cv2.solvePnP(
        model_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Extract robust Euler angles
    angles = rotation_matrix_to_euler_angles(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]

    # Clean up the 180-degree flip ambiguity
    # If the solver flipped the axis, this snaps it back to normal human range
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180

    return roll, pitch, yaw

# def draw_visual_verification(image, rvec, tvec, img_w, img_h):
#     focal_length = img_w
#     center = (img_w / 2, img_h / 2)
#     camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
#     dist_coeffs = np.zeros((4, 1))

#     # Define a 3D Cube around the face space to project as an alignment box
#     box_3d = np.array([
#         (-200, -200, -200), (200, -200, -200), (200, 200, -200), (-200, 200, -200),
#         (-200, -200, 0),    (200, -200, 0),    (200, 200, 0),    (-200, 200, 0)
#     ], dtype="double")

#     # Project the 3D points onto the 2D frame
#     img_points_2d, _ = cv2.projectPoints(box_3d, rvec, tvec, camera_matrix, dist_coeffs)
#     img_points_2d = np.int32(img_points_2d).reshape(-1, 2)

#     # Draw the bounding box lines
#     for i, j in [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]:
#         cv2.line(image, tuple(img_points_2d[i]), tuple(img_points_2d[j]), (0, 255, 0), 2)
    
#     return image

if __name__ == "__main__":
    # Test the head pose estimation and visual verification on a sample image
    import os
    from pathlib import Path

    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent.parent.parent  # Adjust as needed to reach project root
    os.chdir(project_root)

    test_image_path = "public_data/data/jpg/j3nga98r_71_95.jpg"  # Replace with your test image path
    image = cv2.imread(test_image_path)
    img_h, img_w, _ = image.shape

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            rel_dist = calculate_iris_fraction(landmarks, LEFT_IRIS_INDICES, img_w)
            angles = calculate_head_pose(landmarks, img_w, img_h)
            print(f"Relative Iris Distance: {rel_dist}")
            print(f"Head Pose Angles (Roll, Pitch, Yaw): {angles}")
            #verified_image = draw_visual_verification(image.copy(), rvec, tvec, img_w, img_h)
            #cv2.imshow("Head Pose Verification", verified_image)
            #cv2.waitKey(0)
