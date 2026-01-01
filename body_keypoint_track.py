import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle
from typing import List, Tuple, Dict

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

from utils3d import intrinsic_from_fov, mls_smooth_numpy

MEDIAPIPE_POSE_KEYPOINTS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]   # 33

MEDIAPIPE_HAND_KEYPOINTS = [
    "wrist", "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4"
]   # 21

ALL_KEYPOINTS = MEDIAPIPE_POSE_KEYPOINTS + ['left_' + s for s in MEDIAPIPE_HAND_KEYPOINTS] + ['right_' + s for s in MEDIAPIPE_HAND_KEYPOINTS]

MEDIAPIPE_POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)]

WEIGHTS = {
    'left_ear': 0.04,
    'right_ear': 0.04,
    'left_shoulder': 0.18,
    'right_shoulder': 0.18,
    'left_elbow': 0.02,
    'right_elbow': 0.02,
    'left_wrist': 0.01,
    'right_wrist': 0.01,
    'left_hip': 0.2,
    'right_hip': 0.2,
    'left_knee': 0.03,
    'right_knee': 0.03,
    'left_ankle': 0.02,
    'right_ankle': 0.02,
}

class BodyKeypointTrack:
    def __init__(self, im_width: int, im_height: int, fov: float, frame_rate: float, *, track_hands: bool = False, model_path: str = 'pose_landmarker_heavy.task', model_complexity=1, smooth_range: float = 0.3, smooth_range_barycenter: float = 1.0):
        self.K = intrinsic_from_fov(fov, im_width, im_height)
        self.im_width, self.im_height = im_width, im_height
        self.frame_delta = 1. / frame_rate

        # Initialize MediaPipe Tasks Pose Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.pose_rvec, self.pose_tvec = None, None
        self.pose_kpts2d = self.pose_kpts3d = None
        self.barycenter_weight = np.array([WEIGHTS.get(kp, 0.) for kp in MEDIAPIPE_POSE_KEYPOINTS])

        # Hand tracking is disabled for now as we don't have the hand task model
        self.track_hands = False 
        if track_hands:
            print("Warning: Hand tracking is currently disabled because the hand model task is missing.")
            
        self.left_hand_rvec, self.left_hand_tvec = None, None
        self.left_hand_kpts2d = self.left_hand_kpts3d = None
        self.right_hand_rvec, self.right_hand_tvec = None, None
        self.right_hand_kpts2d = self.right_hand_kpts3d = None
        
        self.smooth_range = smooth_range
        self.smooth_range_barycenter = smooth_range_barycenter
        self.barycenter_history: List[Tuple[np.ndarray, float]] = []
        self.pose_history: List[Tuple[np.ndarray, float]] = []
        self.left_hand_history: List[Tuple[np.ndarray, float]] = []
        self.right_hand_history: List[Tuple[np.ndarray, float]] = []

    def _get_camera_space_landmarks(self, image_landmarks, world_landmarks, visible, rvec, tvec):
        # get transformation matrix from world coordinate to camera coordinate
        _, rvec, tvec = cv2.solvePnP(world_landmarks[visible], image_landmarks[visible], self.K, np.zeros(5), rvec=rvec, tvec=tvec, useExtrinsicGuess=rvec is not None)
        rmat, _ = cv2.Rodrigues(rvec)
        
        # get camera coordinate of all keypoints
        kpts3d_cam = world_landmarks @ rmat.T + tvec.T

        # force projected x, y to be identical to visibile image_landmarks
        kpts3d_cam_z = kpts3d_cam[:, 2].reshape(-1, 1)
        kpts3d_cam[:, :2] =  (np.concatenate([image_landmarks, np.ones((image_landmarks.shape[0], 1))], axis=1) @ np.linalg.inv(self.K).T * kpts3d_cam_z)[:, :2]
        return kpts3d_cam, rvec, tvec

    def _track_pose(self, image: np.ndarray, t: float):
        self.pose_kpts2d = self.pose_kpts3d = self.barycenter = None

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect
        # Timestamp must be in milliseconds
        results = self.detector.detect_for_video(mp_image, int(t * 1000))

        if not results.pose_landmarks:
            return 

        # Take the first detected person
        pose_landmarks_proto = results.pose_landmarks[0]
        pose_world_landmarks_proto = results.pose_world_landmarks[0]

        image_landmarks = np.array([[lm.x * self.im_width, lm.y * self.im_height] for lm in pose_landmarks_proto])
        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_world_landmarks_proto])
        visible = np.array([lm.visibility > 0.2 for lm in pose_landmarks_proto])

        if visible.sum() < 6:
            return 
        kpts3d, rvec, tvec = self._get_camera_space_landmarks(image_landmarks, world_landmarks, visible, self.pose_rvec, self.pose_tvec)
        if tvec[2] < 0:
            return

        self.pose_kpts2d = image_landmarks
        self.barycenter = np.average(kpts3d, axis=0, weights=self.barycenter_weight)
        self.pose_kpts3d = kpts3d - self.barycenter
        self.pose_rvec, self.pose_tvec = rvec, tvec
        self.barycenter_history.append((self.barycenter, t))
        self.pose_history.append((kpts3d, t))

    def _track_hands(self, image: np.ndarray, t: float):
        # Placeholder for future implementation with HandLandmarker
        pass

    def track(self, image: np.ndarray, frame_t: float):
        self._track_pose(image, frame_t)
        # Hand tracking skipped for now

    def get_smoothed_3d_keypoints(self, query_t: float):
        # Get smoothed barycenter
        barycenter_list = [barycenter for barycenter, t in self.barycenter_history if abs(t - query_t) < self.smooth_range_barycenter]
        barycenter_t = [t for barycenter, t in self.barycenter_history if abs(t - query_t) < self.smooth_range_barycenter]
        if len(barycenter_t) == 0:
            barycenter = np.zeros(3)
        else:
            barycenter = mls_smooth_numpy(barycenter_t, barycenter_list, query_t, self.smooth_range_barycenter)

        # Get smoothed pose keypoints
        pose_kpts3d_list = [kpts3d for kpts3d, t in self.pose_history if abs(t - query_t) < self.smooth_range]
        pose_t = [t for kpts3d, t in self.pose_history if abs(t - query_t) < self.smooth_range]
        pose_kpts3d = None if not any(abs(t - query_t) < self.frame_delta * 0.6  for t in pose_t) else mls_smooth_numpy(pose_t, pose_kpts3d_list, query_t, self.smooth_range)

        all_kpts3d = pose_kpts3d if pose_kpts3d is not None else np.zeros((len(MEDIAPIPE_POSE_KEYPOINTS), 3))
        all_valid = np.full(len(MEDIAPIPE_POSE_KEYPOINTS), pose_kpts3d is not None)

        if self.track_hands:
             # Logic removed for stability until hand model is available
             # Fill with dummy data
            all_kpts3d = np.concatenate([
                all_kpts3d,
                np.zeros((len(MEDIAPIPE_HAND_KEYPOINTS), 3)),
                np.zeros((len(MEDIAPIPE_HAND_KEYPOINTS), 3))
            ], axis=0)

            all_valid = np.concatenate([
                all_valid,
                np.full(len(MEDIAPIPE_HAND_KEYPOINTS), False),
                np.full(len(MEDIAPIPE_HAND_KEYPOINTS), False)
            ], axis=0)
        
        return all_kpts3d, all_valid

    def get_2d_keypoints(self):
        if self.track_hands:
            return self.pose_kpts2d, self.left_hand_kpts2d, self.right_hand_kpts2d
        else:
            return self.pose_kpts2d

def show_annotation(image, kpts3d, valid, intrinsic):
    annotate_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    kpts3d_homo = kpts3d @ intrinsic.T
    kpts2d = kpts3d_homo[:, :2] / kpts3d_homo[:, 2:]
    for a, b in MEDIAPIPE_POSE_CONNECTIONS:
        if valid[a] == 0 or valid[b] == 0:
            continue
        cv2.line(annotate_image, (int(kpts2d[a, 0]), int(kpts2d[a, 1])), (int(kpts2d[b, 0]), int(kpts2d[b, 1])), (0, 255, 0), 1)
    for i in range(kpts2d.shape[0]):
        if valid[i] == 0:
            continue
        cv2.circle(annotate_image, (int(kpts2d[i, 0]), int(kpts2d[i, 1])), 2, (0, 0, 255), -1)
    cv2.imshow('Keypoint annotation', annotate_image)

def test():
    # Deprecated test function, keeping stub
    pass

if __name__ == '__main__':
    test()
