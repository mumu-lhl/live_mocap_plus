import bpy
import pickle
import numpy as np
import os
import sys

# Ensure stdout shows up in the parent terminal
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

INPUT_FILE = os.path.join('tmp', 'bone_animation_data.pkl')

print("-" * 50)
print(f"[Blender] Starting animation application from {INPUT_FILE}")

try:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Animation data file not found: {INPUT_FILE}")

    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)

    fov = data['fov']
    frame_rate = data['frame_rate']
    bone_names = data['bone_names']
    bone_euler_sequence = data['bone_euler_sequence']
    location_sequence = data['location_sequence']
    scale = data['scale']
    all_bone_names = data['all_bone_names']

    print(f"[Blender] Loaded {len(bone_euler_sequence)} frames, {len(bone_names)} bones.")

    # Camera setup
    if 'Camera' in bpy.data.objects:
        cam = bpy.data.objects['Camera']
        cam.location = (0, 0, 0)
        cam.rotation_euler = (np.pi / 2., 0, 0)
        cam.data.angle = fov
    else:
        print("[Blender] Warning: 'Camera' object not found, skipping camera setup.")

    # Frame settings
    bpy.context.scene.render.fps = int(frame_rate)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(bone_euler_sequence)

    # Skeleton finding logic
    skeleton_objs = list(filter(lambda o: o.type == 'ARMATURE', bpy.data.objects))
    if not skeleton_objs:
        raise Exception("No ARMATURE object found in the scene!")
    
    # Try to find a likely candidate, otherwise pick the first
    skeleton = None
    for obj in skeleton_objs:
        if 'mixamo' in obj.name.lower() or 'armature' in obj.name.lower():
            skeleton = obj
            break
    if skeleton is None:
        skeleton = skeleton_objs[0]
    
    print(f"[Blender] Applying animation to skeleton: '{skeleton.name}'")
    
    # Reset transform
    skeleton.location = (0, 0, 0)
    skeleton.rotation_euler = (-np.pi / 2, 0, 0)
    skeleton.scale = (scale, scale, scale)

    # Validation: Check if bones exist
    pose_bones = skeleton.pose.bones
    missing_bones = [b for b in bone_names if b not in pose_bones]
    if missing_bones:
        print(f"[Blender] WARNING: The following bones exist in data but NOT in skeleton: {missing_bones}")
        print(f"[Blender] Available bones in skeleton: {[b.name for b in pose_bones]}")
    
    # Apply animation
    bpy.context.view_layer.objects.active = skeleton
    bpy.ops.object.mode_set(mode='OBJECT') # Ensure we are in object mode

    print("[Blender] Inserting keyframes...")
    for i in range(len(bone_euler_sequence)):
        # Progress log
        if i % 50 == 0:
            print(f"[Blender] Processing frame {i}/{len(bone_euler_sequence)}...")

        # Apply rotations
        for j, b in enumerate(bone_names):
            if b in pose_bones:
                bone = pose_bones[b]
                bone.rotation_mode = 'YXZ'
                bone.rotation_euler = bone_euler_sequence[i][j].tolist()
                bone.keyframe_insert(data_path='rotation_euler', frame=i)
        
        # Apply root location (Global)
        x, y, z = location_sequence[i].tolist()
        skeleton.location = x, z, -y
        skeleton.keyframe_insert(data_path='location', frame=i)

    print("[Blender] Animation application COMPLETED successfully.")
    print("-" * 50)

except Exception as e:
    import traceback
    print("\n" + "!"*50)
    print("[Blender] ERROR OCCURRED during execution:")
    traceback.print_exc()
    print("!"*50 + "\n")
    # Don't exit here so the user can inspect the scene in Blender GUI if needed