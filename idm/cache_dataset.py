import io
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

from tqdm import tqdm


class CacheDataSet(Dataset):
    """Dual-frame IDM dataset.

    Each sample is a (t, t+k) pair from the same episode (k=frame_skip):
        (img_t, depth_t, img_next, depth_next, pos_t, pos_next)

    frame_skip=4 downsamples 30fps → 8fps (aligned with paper).

    Memory optimization: frames are stored as compressed JPEG bytes in RAM
    (~5-8 GB total) instead of raw PIL Images (~112 GB). Decoding happens
    on-the-fly in __getitem__ and is very fast (~1 ms per frame).

    File layout expected under dataset_path:
        {task_name}/episode_{idx}.mp4
        {task_name}/episode_{idx}_depth.mp4
        {task_name}/episode_{idx}_qpos.pt   — tensor [T, 16] float32
    """

    JPEG_QUALITY = 95  # high quality, ~10-15x compression vs raw pixels

    def __init__(self, args, dataset_path, disable_pbar=False, type="train", preprocessor=None, frame_skip=4):
        self.dataset_path = dataset_path
        self.type = type
        self.frame_skip = frame_skip
        self.preprocessor = preprocessor
        if self.preprocessor is not None:
            self.preprocessor.set_augmentation_progress(0)

        self.rgb_bytes = []      # list of lists of JPEG bytes
        self.depth_bytes = []    # list of lists of PNG bytes (lossless for depth)
        self.qpos_data = []      # list of [T, 16] tensors
        self.pair_counts = []    # valid pairs per episode = T - frame_skip

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

                rgb = self._load_rgb_bytes(rgb_path)
                depth = self._load_depth_bytes(depth_path)
                qpos = torch.load(qpos_path, weights_only=True)

                T = min(len(rgb), len(depth), qpos.shape[0])
                if T < 30:
                    print(f"Skipping {rgb_path} — too short ({T} frames)")
                    continue

                self.rgb_bytes.append(rgb[:T])
                self.depth_bytes.append(depth[:T])
                self.qpos_data.append(qpos[:T])
                self.pair_counts.append(max(T - frame_skip, 0))  # (t, t+k) pairs

        self.data_begin = np.array([0] + list(np.cumsum(self.pair_counts[:-1])), dtype=np.int64)
        self.data_end = np.cumsum(self.pair_counts, dtype=np.int64)

        # 8fps 下采样后不再筛选，保留所有帧对
        total_pairs = int(self.data_end[-1]) if len(self.data_end) > 0 else 0
        self.valid_indices = list(range(total_pairs))
        print(f"[CacheDataSet] {type} 模式 (frame_skip={frame_skip}): 保留全部 {total_pairs} 帧对")

    def __len__(self):
        return len(self.valid_indices)

    # ------------------------------------------------------------------
    # internal helpers — store compressed bytes, decode on-the-fly
    # ------------------------------------------------------------------

    def _load_rgb_bytes(self, path):
        """Load RGB video → list of JPEG bytes (compressed in RAM)."""
        cap = cv2.VideoCapture(path)
        frames = []
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY]
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # frame is BGR uint8; encode to JPEG bytes
            ok_enc, buf = cv2.imencode('.jpg', frame, encode_param)
            if ok_enc:
                frames.append(bytes(buf))
        cap.release()
        return frames

    def _load_depth_bytes(self, path):
        """Load depth video → list of PNG bytes (lossless, single channel)."""
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Take R channel as depth, encode lossless as PNG
            depth_arr = frame[:, :, 2]  # [H, W] uint8
            ok_enc, buf = cv2.imencode('.png', depth_arr)
            if ok_enc:
                frames.append(bytes(buf))
        cap.release()
        return frames

    @staticmethod
    def _decode_rgb(jpeg_bytes):
        """JPEG bytes → PIL Image (RGB)."""
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _decode_depth(png_bytes):
        """PNG bytes → PIL Image (L)."""
        buf = np.frombuffer(png_bytes, dtype=np.uint8)
        depth = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)  # [H, W]
        return Image.fromarray(depth, mode='L')

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        ep = int(np.searchsorted(self.data_end, real_idx, side='right'))
        local_t = int(real_idx - self.data_begin[ep])
        local_next = local_t + self.frame_skip

        # decode on-the-fly from compressed bytes
        img_t    = self._decode_rgb(self.rgb_bytes[ep][local_t])
        img_next = self._decode_rgb(self.rgb_bytes[ep][local_next])
        dep_t    = self._decode_depth(self.depth_bytes[ep][local_t])
        dep_next = self._decode_depth(self.depth_bytes[ep][local_next])
        pos_t    = self.qpos_data[ep][local_t]         # [16]
        pos_next = self.qpos_data[ep][local_next]      # [16]

        if self.preprocessor is not None:
            img_t, dep_t, img_next, dep_next, pos_t, pos_next = \
                self.preprocessor.process_pair(img_t, dep_t, img_next, dep_next, pos_t, pos_next)

        return img_t, dep_t, img_next, dep_next, pos_t, pos_next
