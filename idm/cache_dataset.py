import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

from tqdm import tqdm


class CacheDataSet(Dataset):
    """Dataset for inverse dynamics model

    Contents of Dataset:  If args.load_mp4==True, the data_file_path can be an mp4 file or an qpos file. 
    like:  {task_name}/f'episode_{episode_idx}.mp4', {task_name}/f'episode_{episode_idx}_qpos.pt', where:
    - the mp4 is saved as cv2.VideoWriter({task_name}/f'episode_{episode_idx}.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'), fps=30, (width=640, height=720)). The length of the video is episode_len.
    - the 14-dim qpos of each trajectory is in the form of torch.zeros([episode_len, dim=14]).dtype(float32).cpu(), 
    saved with torch.save()
    - the json file is saved as {task_name}.json', which contains the information of the trajectory, but the caption
    is "Random Wing", not specfically processed yet (no use for IDM, but useful for video generation model).
    """
    
    def __init__(self, args, dataset_path, disable_pbar=False, type="train", preprocessor=None):
        self.data = []
        self.dataset_path = dataset_path
        self.type = type
        self.height = 720
        self.width = 640        
        self.video_frames = []
        self.qpos_data = []
        self.video_lengths = []
        self.preprocessor = preprocessor
        if self.preprocessor is not None:
            self.preprocessor.set_augmentation_progress(0)

        for task_name in os.listdir(dataset_path):
            task_path = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_path):
                continue
            for file_name in tqdm(os.listdir(task_path), desc=f"Loading videos from {task_name}", disable=disable_pbar):
                if file_name.endswith('.mp4'):
                    episode_idx = file_name.split('_')[1].split('.')[0]
                    video_path = os.path.join(task_path, file_name)
                    qpos_path = os.path.join(task_path, f'episode_{episode_idx}_qpos.pt')

                    if not os.path.exists(qpos_path):
                        print(f"Skipping {video_path} - no matching qpos file")
                        continue

                    frames = self.get_images(video_path)
                    video_length = len(frames)
                    qpos = torch.load(qpos_path)

                    if video_length < 30:
                        print(f"Skipping {video_path} - too short")
                        continue
                    self.video_frames.append(frames)
                    self.qpos_data.append(qpos)
                    self.video_lengths.append(video_length)
        self.data_begin = np.cumsum([0] + self.video_lengths[:-1])
        self.data_end = np.cumsum(self.video_lengths)

    def __len__(self):
        return self.data_end[-1]

    def get_images(self, video_path):
        cap = cv2.VideoCapture(video_path)
        success = True
        frames = []
        while success:
            success, frame = cap.read()
            if success:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_idx = np.searchsorted(self.data_end, idx, side='right')
        if video_idx < 0 or video_idx >= len(self.video_frames):
            raise IndexError(f"Index {idx} out of bounds")
        local_idx = idx - self.data_begin[video_idx]
        image = self.video_frames[video_idx][local_idx]
        pos = self.qpos_data[video_idx][local_idx]
        if self.preprocessor is not None:
            image = self.preprocessor.process_image(image)
            if np.random.randn() < 0:
                image, pos = self.preprocessor.handle_flip(image, pos)
        return image, pos
