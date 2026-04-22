"""Convert HDF5 episodes to mp4 + qpos.pt format for IDM training.

Layout (aligned with VIDAR paper):
  - head camera (320x240) → resize to 640x480 (top)
  - left camera (320x240) + right camera (320x240) → side-by-side 640x240 (bottom)
  - Concatenated: 640x720

Outputs per episode:
  episode_XXXXXX.mp4        — RGB video (640x720, 30fps)
  episode_XXXXXX_depth.mp4  — Depth video (640x720, 30fps, depth in R channel)
  episode_XXXXXX_qpos.pt    — Joint positions [T, 16] float32
"""

import os
import sys
import glob
import h5py
import io
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def decode_rgb(rgb_bytes):
    """Decode JPEG bytes to numpy array [H, W, 3] uint8 BGR."""
    img = Image.open(io.BytesIO(rgb_bytes))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def compose_rgb_frame(head_rgb, left_rgb, right_rgb):
    """Compose 3-view RGB into 640x720 frame.
    
    head: 320x240 → resize to 640x480 (top)
    left + right: 320x240 each → side-by-side 640x240 (bottom)
    Result: 640x720
    """
    # head: resize 320x240 → 640x480
    head_big = cv2.resize(head_rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
    
    # left + right side by side: each 320x240 → total 640x240
    bottom = np.concatenate([left_rgb, right_rgb], axis=1)  # [240, 640, 3]
    
    # Stack vertically: 640x480 + 640x240 = 640x720
    frame = np.concatenate([head_big, bottom], axis=0)  # [720, 640, 3]
    return frame


def compose_depth_frame(head_depth, left_depth, right_depth):
    """Compose 3-view depth into 640x720 frame.
    
    Depth values are normalized to uint8 per-frame, stored in R channel of BGR.
    """
    # Normalize each depth to 0-255
    def norm_depth(d):
        d = d.astype(np.float32)
        dmin, dmax = d.min(), d.max()
        if dmax - dmin < 1e-6:
            return np.zeros_like(d, dtype=np.uint8)
        return ((d - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    
    head_u8 = norm_depth(head_depth)    # [240, 320]
    left_u8 = norm_depth(left_depth)    # [240, 320]
    right_u8 = norm_depth(right_depth)  # [240, 320]
    
    # head: resize 320x240 → 640x480
    head_big = cv2.resize(head_u8, (640, 480), interpolation=cv2.INTER_LINEAR)
    
    # left + right side by side
    bottom = np.concatenate([left_u8, right_u8], axis=1)  # [240, 640]
    
    # Stack vertically
    depth_gray = np.concatenate([head_big, bottom], axis=0)  # [720, 640]
    
    # Store as BGR with depth in R channel (same as existing dataset format)
    depth_bgr = np.zeros((720, 640, 3), dtype=np.uint8)
    depth_bgr[:, :, 2] = depth_gray  # R channel
    return depth_bgr


def convert_episode(hdf5_path, output_dir, episode_idx, fps=30):
    """Convert one HDF5 episode to mp4 + qpos.pt."""
    f = h5py.File(hdf5_path, 'r')
    
    # qpos: try multiple possible locations
    if 'qpos' in f:
        qpos_raw = f['qpos'][:]
    elif 'robot_state/qpos' in f:
        qpos_raw = f['robot_state/qpos'][:]
    elif 'joint_action/vector' in f:
        qpos_raw = f['joint_action/vector'][:]
    else:
        f.close()
        return 0  # skip this episode
    
    T = qpos_raw.shape[0]
    qpos = torch.from_numpy(qpos_raw.astype(np.float32))  # [T, 16]
    
    # Check all 3 cameras exist
    required_cams = ['observation/head_camera/rgb', 'observation/left_camera/rgb', 'observation/right_camera/rgb',
                     'observation/head_camera/depth', 'observation/left_camera/depth', 'observation/right_camera/depth']
    for cam in required_cams:
        if cam not in f:
            print(f"Skipping {hdf5_path} — missing {cam}")
            f.close()
            return 0
    
    # Ensure T is consistent with camera frames
    T = min(T, f['observation/head_camera/depth'].shape[0])
    qpos = qpos[:T]
    
    # Output paths
    ep_str = f"episode_{episode_idx:06d}"
    rgb_path = os.path.join(output_dir, f"{ep_str}.mp4")
    depth_path = os.path.join(output_dir, f"{ep_str}_depth.mp4")
    qpos_path = os.path.join(output_dir, f"{ep_str}_qpos.pt")
    
    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rgb_writer = cv2.VideoWriter(rgb_path, fourcc, fps, (640, 720))
    depth_writer = cv2.VideoWriter(depth_path, fourcc, fps, (640, 720))
    
    for t in range(T):
        # Decode RGB
        head_rgb = decode_rgb(f['observation/head_camera/rgb'][t])
        left_rgb = decode_rgb(f['observation/left_camera/rgb'][t])
        right_rgb = decode_rgb(f['observation/right_camera/rgb'][t])
        
        rgb_frame = compose_rgb_frame(head_rgb, left_rgb, right_rgb)
        rgb_writer.write(rgb_frame)
        
        # Depth
        head_depth = f['observation/head_camera/depth'][t]
        left_depth = f['observation/left_camera/depth'][t]
        right_depth = f['observation/right_camera/depth'][t]
        
        depth_frame = compose_depth_frame(head_depth, left_depth, right_depth)
        depth_writer.write(depth_frame)
    
    rgb_writer.release()
    depth_writer.release()
    
    # Save qpos
    torch.save(qpos, qpos_path)
    
    f.close()
    return T


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to mp4 + qpos.pt")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to HDF5 data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for mp4/qpos files")
    parser.add_argument("--task_name", type=str, default="robotwin", help="Task folder name in output")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to convert (for testing)")
    args = parser.parse_args()
    
    # Create output directory: output_dir/task_name/
    task_output = os.path.join(args.output_dir, args.task_name)
    os.makedirs(task_output, exist_ok=True)
    
    # Find all HDF5 files
    hdf5_files = sorted(glob.glob(os.path.join(args.input_dir, "episode*.hdf5")))
    if args.max_episodes:
        hdf5_files = hdf5_files[:args.max_episodes]
    
    print(f"Found {len(hdf5_files)} episodes in {args.input_dir}")
    print(f"Output: {task_output}")
    print(f"FPS: {args.fps}")
    
    total_frames = 0
    converted = 0
    skipped = 0
    for i, hdf5_path in enumerate(tqdm(hdf5_files, desc="Converting")):
        try:
            T = convert_episode(hdf5_path, task_output, converted, fps=args.fps)
            if T > 0:
                total_frames += T
                converted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\nSkipping {hdf5_path}: {e}")
            skipped += 1
    
    print(f"\nDone! Converted {converted} episodes ({skipped} skipped), {total_frames} total frames")
    print(f"Output: {task_output}")


if __name__ == "__main__":
    main()
