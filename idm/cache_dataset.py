import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

from tqdm import tqdm


class CacheDataSet(Dataset):
    """Dual-frame IDM dataset.

    Each sample is a consecutive (t, t+1) pair from the same episode:
        (img_t, depth_t, img_next, depth_next, pos_t, pos_next)

    File layout expected under dataset_path:
        {task_name}/episode_{idx}.mp4
        {task_name}/episode_{idx}_depth.mp4
        {task_name}/episode_{idx}_qpos.pt   — tensor [T, 16] float32

    The last frame of every episode is excluded (no t+1 partner).
    """

    def __init__(self, args, dataset_path, disable_pbar=False, type="train", preprocessor=None):
        self.dataset_path = dataset_path
        self.type = type
        self.preprocessor = preprocessor
        if self.preprocessor is not None:
            self.preprocessor.set_augmentation_progress(0)

        self.rgb_frames = []     # list of lists of PIL Images
        self.depth_frames = []   # list of lists of PIL Images (single-channel)
        self.qpos_data = []      # list of [T, 16] tensors
        self.pair_counts = []    # valid pairs per episode = T - 1

        for task_name in os.listdir(dataset_path):
            task_path = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_path):
                continue
            for file_name in tqdm(os.listdir(task_path), desc=f"Loading {task_name}", disable=disable_pbar):
                if not (file_name.endswith('.mp4') and '_depth' not in file_name):
                    continue

                episode_idx = file_name.replace('episode_', '').replace('.mp4', '')
                rgb_path = os.path.join(task_path, file_name)
                depth_path = os.path.join(task_path, f'episode_{episode_idx}_depth.mp4')
                qpos_path = os.path.join(task_path, f'episode_{episode_idx}_qpos.pt')

                if not os.path.exists(qpos_path):
                    print(f"Skipping {rgb_path} — no qpos file")
                    continue
                if not os.path.exists(depth_path):
                    print(f"Skipping {rgb_path} — no depth file")
                    continue

                rgb = self._load_rgb(rgb_path)
                depth = self._load_depth(depth_path)
                qpos = torch.load(qpos_path, weights_only=True)

                T = min(len(rgb), len(depth), qpos.shape[0])
                if T < 30:
                    print(f"Skipping {rgb_path} — too short ({T} frames)")
                    continue

                self.rgb_frames.append(rgb[:T])
                self.depth_frames.append(depth[:T])
                self.qpos_data.append(qpos[:T])
                self.pair_counts.append(T - 1)   # last frame has no t+1

        self.data_begin = np.array([0] + list(np.cumsum(self.pair_counts[:-1])), dtype=np.int64)
        self.data_end = np.cumsum(self.pair_counts, dtype=np.int64)

    def __len__(self):
        return int(self.data_end[-1]) if len(self.data_end) > 0 else 0

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_rgb(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames

    def _load_depth(self, path):
        """Load depth video. Stored as BGR mp4 where depth = R channel (uint8 proxy)."""
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Take only the R channel as a single-channel grayscale depth image
            depth_arr = frame[:, :, 2]  # [H, W] uint8
            frames.append(Image.fromarray(depth_arr, mode='L'))
        cap.release()
        return frames

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        ep = int(np.searchsorted(self.data_end, idx, side='right'))
        local_t = int(idx - self.data_begin[ep])     # frame index t within episode

        img_t    = self.rgb_frames[ep][local_t]
        img_next = self.rgb_frames[ep][local_t + 1]
        dep_t    = self.depth_frames[ep][local_t]
        dep_next = self.depth_frames[ep][local_t + 1]
        pos_t    = self.qpos_data[ep][local_t]        # [16]
        pos_next = self.qpos_data[ep][local_t + 1]   # [16]

        if self.preprocessor is not None:
            img_t, dep_t, img_next, dep_next, pos_t, pos_next = \
                self.preprocessor.process_pair(img_t, dep_t, img_next, dep_next, pos_t, pos_next)

        return img_t, dep_t, img_next, dep_next, pos_t, pos_next
