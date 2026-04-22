import os
import numpy as np
import torch
import wandb
import argparse
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import cv2

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from idm.cache_dataset import CacheDataSet
from idm.idm import IDM, OUTPUT_DIM
from idm.preprocessor import DinoPreprocessor
from idm.utils import seed_torch

# Tolerance per joint dimension used for "close enough" accuracy metric.
# From paper: max infinity norm error < 0.06 for joints, < 0.6 for grippers.
# Layout: [left_arm(7), left_grip(1), right_arm(7), right_grip(1)]
_CLOSE_LIMIT = torch.tensor([
    0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,   # left arm
    0.6,                                           # left gripper
    0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,   # right arm
    0.6,                                           # right gripper
])


def parse_args():
    parser = argparse.ArgumentParser(description="Train IDM (dual-frame, predict pos_{t+1})")
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--use_transform", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--num_iterations", type=int, default=150000)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--ratio_eval", type=float, default=0.05)
    parser.add_argument("--model_name", type=str, default="mask")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--test_dataset_path", nargs="+", default=[])
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False, help="Only run test eval, skip train/val loading")
    return parser.parse_args()


def collate_fn(batch):
    """batch: list of (img_t, dep_t, img_next, dep_next, pos_t, pos_next)"""
    img_t, dep_t, img_next, dep_next, pos_t, pos_next = zip(*batch)
    return (
        torch.stack(img_t),     # [B, 3, H, W]
        torch.stack(dep_t),     # [B, 1, H, W]
        torch.stack(img_next),  # [B, 3, H, W]
        torch.stack(dep_next),  # [B, 1, H, W]
        torch.stack(pos_t),     # [B, 16]
        torch.stack(pos_next),  # [B, 16]
    )


def get_data_generator(dataloader):
    while True:
        for data in dataloader:
            yield data


def save_model(accelerator, net, optimizer, step, save_path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": accelerator.unwrap_model(net).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        }, save_path)
    accelerator.wait_for_everyone()


def is_close(pos_true, pos_pred, device):
    """Check per-sample whether predicted pos is within tolerance of ground truth."""
    limit = _CLOSE_LIMIT.to(device)
    if pos_true.dim() == 1:
        return torch.all(torch.abs(pos_true - pos_pred) < limit)
    return torch.all(torch.abs(pos_true - pos_pred) < limit, dim=1)


def eval(accelerator, net, dataloader, loss_fn, step, mode='val', save_dir='output'):
    os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    net.eval()
    first_batch = True

    with torch.no_grad():
        eval_loss = 0.0
        eval_l1_error = 0.0
        total_correct = 0
        total_samples = 0

        for img_t, dep_t, img_next, dep_next, pos_t, pos_next in tqdm(dataloader, disable=not accelerator.is_main_process):
            pos_pred = net(img_t, dep_t, img_next, dep_next, pos_t)

            # --- Visualize mask on first batch ---
            if first_batch and accelerator.is_main_process:
                try:
                    _, masks = accelerator.unwrap_model(net).model.forward(
                        img_t, dep_t, img_next, dep_next, pos_t, return_mask=True)
                    if isinstance(masks, tuple):
                        mask_t_vis, mask_next_vis = masks
                    else:
                        mask_t_vis = masks
                        mask_next_vis = None

                    for tag, img_vis, mask_vis in [("t", img_t, mask_t_vis), ("next", img_next, mask_next_vis)]:
                        if mask_vis is None:
                            continue
                        # Denormalize RGB
                        rgb = img_vis[0].detach().cpu().numpy().transpose(1, 2, 0)
                        rgb = rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

                        m = mask_vis[0, 0].detach().cpu().numpy()
                        m_uint8 = np.clip(m * 255, 0, 255).astype(np.uint8)

                        # Mask overlay: green tint on masked region
                        overlay = rgb.copy()
                        overlay[m > 0.5, 1] = np.clip(overlay[m > 0.5, 1].astype(int) + 80, 0, 255).astype(np.uint8)

                        # Save: mask alone + overlay
                        cv2.imwrite(os.path.join(save_dir, f'mask_{tag}_{mode}_{step}.png'), m_uint8)
                        cv2.imwrite(os.path.join(save_dir, f'mask_overlay_{tag}_{mode}_{step}.png'), overlay[:, :, ::-1])

                    # Log mask stats
                    mask_mean = mask_t_vis[0].mean().item()
                    mask_ratio = (mask_t_vis[0] > 0.5).float().mean().item()
                    print(f"  mask stats: mean={mask_mean:.4f}, ratio(>0.5)={mask_ratio:.2%}")
                except Exception as e:
                    print(f"  [mask vis skipped: {e}]")

            pos_next_g = accelerator.gather(pos_next)
            pos_pred_g = accelerator.gather(pos_pred)
            pos_t_g    = accelerator.gather(pos_t)

            if accelerator.is_main_process:
                B = pos_next_g.shape[0]
                total_samples += B

                l1 = torch.abs(pos_next_g - pos_pred_g).mean(dim=1)
                eval_l1_error += l1.sum().item()

                close = is_close(pos_next_g, pos_pred_g, pos_next_g.device)
                total_correct += close.sum().item()

                loss = loss_fn(pos_pred_g, pos_next_g)
                eval_loss += loss.item() * B

                if first_batch:
                    # Save sample RGB frame
                    sample_rgb = img_t[0].detach().cpu().numpy()
                    sample_rgb = np.transpose(sample_rgb, (1, 2, 0))
                    sample_rgb = sample_rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    sample_rgb = np.clip(sample_rgb * 255, 0, 255).astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(os.path.join(save_dir, f'image_{mode}_{step}.png'), sample_rgb)

                    fmt = lambda v: ', '.join(f'{x:.4f}' for x in v)
                    joint_err = (pos_pred_g[0] - pos_next_g[0]).abs().cpu()

                    print(f"\npos_t      [0]: [{fmt(pos_t_g[0].cpu())}]")
                    print(f"pos_next   [0]: [{fmt(pos_next_g[0].cpu())}]")
                    print(f"pos_pred   [0]: [{fmt(pos_pred_g[0].cpu())}]")
                    print(f"joint_err  [0]: [{fmt(joint_err)}]  mean={joint_err.mean():.4f}  max={joint_err.max():.4f}")
                    print(f"Correct?  {close[0].item()}")
                    first_batch = False

        if accelerator.is_main_process:
            eval_loss      /= total_samples
            eval_l1_error  /= total_samples
            correct_rate    = total_correct / total_samples

            print(f"{mode} loss={eval_loss:.4f}  l1={eval_l1_error:.4f}  acc={correct_rate:.4f}")
            if wandb.run is not None:
                wandb.log({
                    f"{mode}_loss": eval_loss,
                    f"{mode}_l1_error": eval_l1_error,
                    f"{mode}_correct_rate": correct_rate,
                }, step=step)

    net.train()
    accelerator.wait_for_everyone()


def main(args):
    seed_torch(1234)
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    num_gpus = max(torch.cuda.device_count(), 1)
    save_dir = os.path.join(args.save_dir, args.run_name)

    if accelerator.is_main_process and not args.eval_only:
        os.makedirs(save_dir, exist_ok=True)
        wandb.init(project=f"IDM_{args.model_name}", mode=args.wandb_mode,
                   config=args.__dict__, name=args.run_name)

    if accelerator.is_main_process:
        print(args.__dict__)

    preprocessor = DinoPreprocessor(args)

    # --test_only: skip train/val, only load test datasets
    if args.test_only:
        if not args.load_from:
            raise ValueError("--test_only requires --load_from")

        test_datasets = [
            CacheDataSet(args, dataset_path=p,
                         disable_pbar=not accelerator.is_main_process,
                         type="test", preprocessor=preprocessor)
            for p in args.test_dataset_path
        ]
        if accelerator.is_main_process:
            print(f"test={[len(d) for d in test_datasets]}")

        dl_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         collate_fn=collate_fn, prefetch_factor=args.prefetch_factor)
        test_dataloaders = [DataLoader(d, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **dl_kwargs) for d in test_datasets]

        net = IDM(model_name=args.model_name, output_dim=OUTPUT_DIM)
        loss_fn = nn.SmoothL1Loss()
        ckpt = torch.load(args.load_from, map_location='cpu')
        net.load_state_dict(ckpt["model_state_dict"])
        if accelerator.is_main_process:
            print(f"Loaded checkpoint from {args.load_from}")

        net = accelerator.prepare(net)
        test_dataloaders = [accelerator.prepare(dl) for dl in test_dataloaders]

        preprocessor.use_transform = False
        for i, dl in enumerate(test_dataloaders):
            eval(accelerator, net, dl, loss_fn, 0, mode=f'test{i}', save_dir=save_dir)
        return

    dataset = CacheDataSet(args, dataset_path=args.dataset_path,
                           disable_pbar=not accelerator.is_main_process,
                           preprocessor=preprocessor)
    test_datasets = [
        CacheDataSet(args, dataset_path=p,
                     disable_pbar=not accelerator.is_main_process,
                     type="test", preprocessor=preprocessor)
        for p in args.test_dataset_path
    ]

    dataset_size     = len(dataset)
    val_size         = min(int(args.ratio_eval * dataset_size), 10000)
    train_size       = dataset_size - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    if accelerator.is_main_process:
        print(f"train={train_size}  val={val_size}  test={[len(d) for d in test_datasets]}")

    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                     collate_fn=collate_fn, prefetch_factor=args.prefetch_factor)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    val_dataloader   = DataLoader(val_ds,   batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **dl_kwargs)
    test_dataloaders = [DataLoader(d, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **dl_kwargs) for d in test_datasets]

    net = IDM(model_name=args.model_name, output_dim=OUTPUT_DIM)
    optimizer = AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    loss_fn = nn.SmoothL1Loss()  # Huber loss (same as paper)

    # Cosine LR with linear warmup
    if args.lr_scheduler == "cosine":
        warmup_steps = int(0.1 * args.num_iterations)
        def lr_lambda(step):
            step = step // num_gpus
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, args.num_iterations - warmup_steps)
            return 0.5 * (np.cos(progress * np.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    start_step = 0
    if args.load_from and os.path.isfile(args.load_from):
        ckpt = torch.load(args.load_from, map_location='cpu')
        net.load_state_dict(ckpt["model_state_dict"])
        if not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt["step"]
            if scheduler is not None:
                for _ in range(start_step):
                    scheduler.step()
        if accelerator.is_main_process:
            print(f"Loaded checkpoint from {args.load_from} (step {start_step})")
    elif args.eval_only:
        raise ValueError("--eval_only requires --load_from")

    net, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        net, optimizer, train_dataloader, val_dataloader)
    test_dataloaders = [accelerator.prepare(dl) for dl in test_dataloaders]
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    if args.eval_only:
        preprocessor.use_transform = False
        eval(accelerator, net, val_dataloader, loss_fn, 0, mode='val', save_dir=save_dir)
        for i, dl in enumerate(test_dataloaders):
            eval(accelerator, net, dl, loss_fn, 0, mode=f'test{i}', save_dir=save_dir)
        return

    net.train()
    train_gen = get_data_generator(train_dataloader)

    pbar = tqdm(range(start_step, args.num_iterations), disable=not accelerator.is_main_process)
    for step in pbar:
        img_t, dep_t, img_next, dep_next, pos_t, pos_next = next(train_gen)

        # Model predicts pos_{t+1} (absolute position)
        pos_pred = net(img_t, dep_t, img_next, dep_next, pos_t)

        # Huber loss: predict absolute pos_{t+1}
        loss = loss_fn(pos_pred, pos_next)

        optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(accelerator.unwrap_model(net).parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            batch_acc = is_close(pos_next, pos_pred, pos_next.device).float().mean().item()
            lr_now = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.2e}",
                             lr=f"{lr_now:.2e}", acc=f"{batch_acc:.3f}")
            if step % 10 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": lr_now,
                    "batch_accuracy": batch_acc,
                }, step=step)

        if (step + 1) % args.eval_interval == 0:
            preprocessor.use_transform = False
            eval(accelerator, net, val_dataloader, loss_fn, step + 1, mode='val', save_dir=save_dir)
            for i, dl in enumerate(test_dataloaders):
                eval(accelerator, net, dl, loss_fn, step + 1, mode=f'test{i}', save_dir=save_dir)
            preprocessor.use_transform = args.use_transform
            save_model(accelerator, net, optimizer, step + 1,
                       os.path.join(save_dir, f"{step + 1}.pt"))

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
