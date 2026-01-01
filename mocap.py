#####################################################################################
# Single View Human Motion Capture, Based on Mediapipe & OpenCV & PyTorch
# 
# Author: Ruicheng Wang
# License: Apache License 2.0
#####################################################################################
import os
import shutil
import argparse
import pickle
import subprocess

import numpy as np
import cv2
import torch
from tqdm import tqdm

from body_keypoint_track import BodyKeypointTrack, show_annotation
from skeleton_ik_solver import SkeletonIKSolver


def rotate_image(image, angle_code):
    if angle_code == 0: return image
    if angle_code == 1: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle_code == 2: return cv2.rotate(image, cv2.ROTATE_180)
    if angle_code == 3: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def get_user_rotation(cap):
    """
    Interactive rotation calibration
    """
    current_rot = 0
    print("\n[Calibration] Press 'R' to rotate, 'Space' to confirm, 'Esc' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        display_frame = rotate_image(frame, current_rot)
        
        # Scale for preview
        h, w = display_frame.shape[:2]
        if h > 800:
            scale = 800 / h
            display_frame = cv2.resize(display_frame, (int(w*scale), int(h*scale)))

        cv2.putText(display_frame, f"Rot: {current_rot*90} deg", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Calibration', display_frame)
        
        key = cv2.waitKey(0)
        if key == ord('r') or key == ord('R'):
            current_rot = (current_rot + 1) % 4
        elif key == 32: # Space
            cv2.destroyWindow('Calibration')
            return current_rot
        elif key == 27: # Esc
            print('Cancelled by user. Exit.')
            exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', type=str, help='Path to rigged model Blender file. eg. c:\\tmp\\model.blend')
    parser.add_argument('--video', type=str, help='Path to video file. eg. c:\\tmp\\video.mp4')
    parser.add_argument('--track_hands', action='store_true', help='Enable hand tracking')
    parser.add_argument('--rotation', type=int, default=None, choices=[0, 1, 2, 3], help='Rotation code: 0: 0, 1: 90CW, 2: 180, 3: 90CCW')
    parser.add_argument('--no_gui', action='store_true', help='Disable GUI preview')

    args = parser.parse_args()
    FOV = np.pi / 3

    # Call blender to export skeleton
    os.makedirs('tmp', exist_ok=True)
    print("Export skeleton...")
    skeleton_path = os.path.join('tmp', 'skeleton')
    if os.path.exists(skeleton_path):
        shutil.rmtree(skeleton_path)
    
    # Use subprocess to capture output for better debugging
    cmd = f"blender {args.blend} --background --python export_skeleton.py"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0 or not os.path.exists(skeleton_path):
        print("\n" + "="*40)
        print("ERROR: Blender export failed.")
        print("="*40)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("="*40 + "\n")
        raise Exception("Skeleton export failed. See output above for details.")

    # Open the video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise Exception("Video capture failed")
    
    # Handle Rotation
    if args.rotation is not None:
        rotation = args.rotation
    elif not args.no_gui:
        rotation = get_user_rotation(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        rotation = 0

    raw_width, raw_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotation % 2 == 1:
        frame_width, frame_height = raw_height, raw_width
    else:
        frame_width, frame_height = raw_width, raw_height

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the body keypoint tracker
    body_keypoint_track = BodyKeypointTrack(
        im_width=frame_width,
        im_height=frame_height,
        fov=FOV,
        frame_rate=frame_rate,
        track_hands=args.track_hands,
        smooth_range=10 * (1 / frame_rate),
        smooth_range_barycenter=30 * (1 / frame_rate),
        model_path='pose_landmarker_heavy.task'
    )

    # Initialize the skeleton IK solver
    skeleton_ik_solver = SkeletonIKSolver(
        model_path='tmp/skeleton',
        track_hands=args.track_hands,
        smooth_range=15 * (1 / frame_rate),
    )

    bone_euler_sequence, scale_sequence, location_sequence = [], [], []

    frame_t = 0.0
    frame_i = 0
    bar = tqdm(total=total_frames, desc='Running...')
    while cap.isOpened():
        # Get the frame image
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rotation
        frame = rotate_image(frame, rotation)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the body keypoints
        body_keypoint_track.track(frame, frame_t)
        kpts3d, valid = body_keypoint_track.get_smoothed_3d_keypoints(frame_t)

        # Solve the skeleton IK
        skeleton_ik_solver.fit(torch.from_numpy(kpts3d).float(), torch.from_numpy(valid).bool(), frame_t)

        # Get the skeleton pose
        bone_euler = skeleton_ik_solver.get_smoothed_bone_euler(frame_t)
        location = skeleton_ik_solver.get_smoothed_location(frame_t)
        scale = skeleton_ik_solver.get_scale()

        # Convert to numpy immediately to avoid pickling torch tensors
        bone_euler_sequence.append(bone_euler.detach().cpu().numpy())
        location_sequence.append(location.detach().cpu().numpy())
        scale_sequence.append(scale)

        # Show keypoints tracking result
        if not args.no_gui:
            show_annotation(frame, kpts3d, valid, body_keypoint_track.K)
            if cv2.waitKey(1) == 27:
                print('Cancelled by user. Exit.')
                exit()

        frame_i += 1
        frame_t += 1.0 / frame_rate
        bar.update(1)

    # Save animation result...
    print("Save animation result...")
    print(f"[DEBUG] Total frames generated: {len(bone_euler_sequence)}")
    print(f"[DEBUG] Optimizable bones: {len(skeleton_ik_solver.optimizable_bones)}")
    if len(bone_euler_sequence) > 0:
        print(f"[DEBUG] Sample euler (frame 0, bone 0): {bone_euler_sequence[0][0]}")
    
    with open('tmp/bone_animation_data.pkl', 'wb') as fp:
        pickle.dump({
            'fov': FOV,
            'frame_rate': frame_rate,
            'bone_names': skeleton_ik_solver.optimizable_bones,
            'bone_euler_sequence': bone_euler_sequence,
            'location_sequence': location_sequence,
            'scale': np.mean(scale_sequence),
            'all_bone_names': skeleton_ik_solver.all_bone_names
        }, fp)

    # Open blender and apply the animation
    print("Open blender and apply animation...")
    
    cmd = ["blender", args.blend, "--python", "apply_animation.py"]
    proc = subprocess.Popen(cmd)
    proc.wait()


if __name__ == '__main__':
    main()