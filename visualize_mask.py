"""Load a checkpoint and visualize the learned mask on a few samples."""
import os
import sys
import argparse
import torch
import numpy as np
import cv2

from idm.idm import IDM, OUTPUT_DIM
from idm.cache_dataset import CacheDataSet
from idm.preprocessor import DinoPreprocessor
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--dataset_path", type=str, default="data/assets/train_sigleview")
    parser.add_argument("--save_dir", type=str, default="output/mask_vis")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to visualize")
    parser.add_argument("--model_name", type=str, default="mask")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build preprocessor (no augmentation) ---
    class FakeArgs:
        use_transform = False
    preprocessor = DinoPreprocessor(FakeArgs())

    # --- Load dataset ---
    class DataArgs:
        use_transform = False
        dataset_path = args.dataset_path
    dataset = CacheDataSet(DataArgs(), dataset_path=args.dataset_path,
                           type="test", preprocessor=preprocessor)

    def collate_fn(batch):
        img_t, dep_t, img_next, dep_next, pos_t, pos_next = zip(*batch)
        return (torch.stack(img_t), torch.stack(dep_t),
                torch.stack(img_next), torch.stack(dep_next),
                torch.stack(pos_t), torch.stack(pos_next))

    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)

    # --- Load model ---
    net = IDM(model_name=args.model_name, output_dim=OUTPUT_DIM)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device)
    net.eval()

    # --- Get one batch ---
    img_t, dep_t, img_next, dep_next, pos_t, pos_next = next(iter(loader))
    img_t, dep_t = img_t.to(device), dep_t.to(device)
    img_next, dep_next = img_next.to(device), dep_next.to(device)
    pos_t = pos_t.to(device)

    with torch.no_grad():
        _, masks = net.model.forward(img_t, dep_t, img_next, dep_next, pos_t, return_mask=True)

    if isinstance(masks, tuple):
        mask_t, mask_next = masks
    else:
        mask_t = masks
        mask_next = None

    # --- Denormalize and save ---
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(min(args.num_samples, img_t.shape[0])):
        for tag, img_tensor, mask_tensor in [("t", img_t, mask_t), ("next", img_next, mask_next)]:
            if mask_tensor is None:
                continue

            # RGB
            rgb = img_tensor[i].cpu().numpy().transpose(1, 2, 0)
            rgb = rgb * std + mean
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

            # Mask
            m = mask_tensor[i, 0].cpu().numpy()
            m_uint8 = np.clip(m * 255, 0, 255).astype(np.uint8)

            # Overlay: green tint on masked region
            overlay = rgb.copy()
            mask_bool = m > 0.5
            overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1].astype(int) + 80, 0, 255).astype(np.uint8)

            # Masked image (background zeroed out)
            masked_rgb = rgb.copy()
            masked_rgb[~mask_bool] = 0

            # Save
            cv2.imwrite(os.path.join(args.save_dir, f"sample{i}_{tag}_rgb.png"), rgb[:, :, ::-1])
            cv2.imwrite(os.path.join(args.save_dir, f"sample{i}_{tag}_mask.png"), m_uint8)
            cv2.imwrite(os.path.join(args.save_dir, f"sample{i}_{tag}_overlay.png"), overlay[:, :, ::-1])
            cv2.imwrite(os.path.join(args.save_dir, f"sample{i}_{tag}_masked.png"), masked_rgb[:, :, ::-1])

            ratio = mask_bool.mean()
            print(f"  sample{i}_{tag}: mask_mean={m.mean():.4f}, ratio(>0.5)={ratio:.2%}")

    print(f"\nSaved {args.num_samples} samples to {args.save_dir}")


if __name__ == "__main__":
    main()
