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
from idm.idm import *
from idm.preprocessor import DinoPreprocessor
from idm.utils import seed_torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train IDM")
    parser.add_argument("--load_from", type=str, default=None, help="Load from path")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Wandb mode")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--mask_weight", type=float, default=1e-3, help="Mask weight")
    parser.add_argument("--use_transform", action="store_true", default=False, help="Use transform")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loading workers")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Number of batches to prefetch")
    parser.add_argument("--dataset_path", type=str, default="", help="Path of the dataset")
    parser.add_argument("--num_iterations", type=int, default=150000, help="Number of iterations")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Intervals of evaluation. ")
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Run name")
    parser.add_argument("--save_dir", type=str, default="output", help="Save dir")
    parser.add_argument("--ratio_eval", type=float, default=0.05, help="Ratio of data for validation, but eval_dataset_size is at most 10000")
    parser.add_argument("--model_name", type=str, default="mask", help="Choose a suitable model.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "cosine"], help="Learning rate scheduler type")
    parser.add_argument("--test_dataset_path", nargs="+", default=[], help="Path of the test dataset")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Only run evaluation on val and test sets")
    parser.add_argument("--use_normalization", action="store_true", default=False, help="Use mean/std normalization")
    parser.add_argument("--load_mp4", action="store_true", default=True, help="load the data in mp4 format to save memory")
    args = parser.parse_args()
    return args


def collate_fn(batch):
    # batch is a list of tuples (pos, image), pos is [B, 14], image is [B, 3, 518, 518] tensor
    # image is a pil image: [H, W, C]
    # concatenate all pos and images from list to tensor
    images, pos = zip(*batch)
    # preprocess images
    images = torch.stack(images)
    pos = torch.stack(pos)  # [B, 14]
    return images, pos


def get_data_generator(dataloader):
    while True:
        for data in dataloader:
            yield data


def save_model(accelerator: Accelerator, net: torch.nn.Module, optimizer: torch.optim.Optimizer, step, save_path):
    accelerator.wait_for_everyone()
    save_dir = os.path.dirname(save_path)
    if accelerator.is_main_process:
        try:
            os.makedirs(save_dir, exist_ok=True)
            if not os.access(save_dir, os.W_OK):
                print(f"Warning: No write permission for directory {save_dir}")
                return

            state_dict = {
                "model_state_dict": accelerator.unwrap_model(net).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            }
            torch.save(state_dict, save_path)
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    accelerator.wait_for_everyone()


def is_close(pos, output):
    limit = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]).to(pos.device)
    # gripper:
    limit[6] = 0.5
    limit[13] = 0.5
    # Handle both single samples and batches
    if pos.dim() == 1:
        return torch.all(torch.abs(pos - output) < limit)
    else:
        return torch.all(torch.abs(pos - output) < limit, dim=1)


def eval(accelerator: Accelerator, net: torch.nn.Module, dataloader: DataLoader, loss_fn, step, use_normalization, mode='val', save_dir='output'):
    os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    net.eval()
    first_batch = True
    with torch.no_grad():
        eval_loss = 0
        eval_l1_error = 0
        total_correct = 0
        total_samples = 0

        # Get learning dimensions mask from loss function
        learning_mask = loss_fn.learning_mask.to(accelerator.device) if hasattr(loss_fn, 'learning_mask') else torch.ones(14, dtype=torch.bool).to(accelerator.device)
        active_dims = learning_mask.sum().item()
        
        for images, pos in tqdm(dataloader, disable=not accelerator.is_main_process):
            pos = accelerator.gather(pos)
            output = net(images, return_mask=True)
            if isinstance(output, tuple):
                output, mask = output
                output = accelerator.gather(output)
                mask = accelerator.gather(mask)
            else:
                output = accelerator.gather(output)
                mask = None

            if accelerator.is_main_process:
                # Only compute metrics for learned dimensions
                masked_abs_error = torch.abs(pos - output) * learning_mask.float()
                eval_l1_error += (masked_abs_error.sum(dim=1) / active_dims).sum().item() if active_dims > 0 else 0
                
                # For is_close calculation, we only check dimensions we're learning
                if active_dims > 0:
                    is_close_mask = torch.abs(pos - output) < torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5]).to(pos.device)
                    # A sample is correct only if all learned dimensions are close
                    correct_samples = is_close_mask[:, learning_mask].float()
                    correct_samples = torch.all(correct_samples, dim=1)
                    total_correct += correct_samples.sum().item()

                total_samples += len(pos)
                
                if first_batch:
                    sample_image = images[0].detach().cpu().numpy()
                    sample_image = np.transpose(sample_image, (1, 2, 0))
                    sample_image *= np.array([0.229, 0.224, 0.225])
                    sample_image += np.array([0.485, 0.456, 0.406])
                    sample_image = np.clip(sample_image, 0, 1)
                    sample_image = (sample_image * 255).astype(np.uint8)[:, :, [2, 1, 0]]
                    cv2.imwrite(os.path.join(save_dir, f'image_{mode}_{step}.png'), sample_image)

                    if mask is not None:
                        sample_mask = mask[0].detach().cpu().numpy()
                        sample_mask = np.transpose(sample_mask, (1, 2, 0))
                        sample_mask = np.where(sample_mask >= 0.5, sample_image, 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(save_dir, f'mask_{mode}_{step}.png'), sample_mask)

                    sample_pos = pos[0].detach().cpu().numpy()
                    sample_output = output[0].detach().cpu().numpy()
                    is_correct = is_close(pos[0], output[0]).item()

                    formatted_pos = ', '.join([f"{val:.4f}" for val in sample_pos])
                    formatted_output = ', '.join([f"{val:.4f}" for val in sample_output])
                    
                    print(f"\nSample pos: [{formatted_pos}]")
                    print(f"Sample output: [{formatted_output}]")
                    print(f"Correct?: {is_correct}")
                    first_batch = False
                
                # For loss calculation, normalize pos if normalization is used
                if use_normalization:
                    loss = loss_fn(net.normalize(output), net.normalize(pos))
                else:
                    loss = loss_fn(output, pos)
                eval_loss += loss.item() * len(pos)

        if accelerator.is_main_process:
            eval_loss /= total_samples
            eval_l1_error /= total_samples
            correct_rate = total_correct / total_samples if total_samples > 0 else 0.0
            
            # Print results instead of logging to wandb in eval-only mode
            print(f"{mode}_loss: {eval_loss:.4f}, {mode}_l1_error: {eval_l1_error:.4f}, correct_rate: {correct_rate:.4f}")
            
            # Only log to wandb if it's initialized
            if wandb.run is not None:
                wandb.log({
                    f"{mode}_loss": eval_loss, 
                    f"{mode}_l1_error": eval_l1_error, 
                    f"{mode}_correct_rate": correct_rate
                }, step=step)
    
    net.train()
    accelerator.wait_for_everyone()


def main(args):
    seed_torch(1234)
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    num_gpus = torch.cuda.device_count()
    save_dir = os.path.join(args.save_dir, args.run_name)
    
    # Initialize wandb only if not in eval mode
    if accelerator.is_main_process and not args.eval_only:
        os.makedirs(save_dir, exist_ok=True)
        wandb.init(project=f"IDM_{args.model_name}", mode=args.wandb_mode, config=args.__dict__, name=args.run_name)
    
    if accelerator.is_main_process:
        print(f"{args.__dict__}")

    # Initialize preprocessor
    preprocessor = DinoPreprocessor(args)
    
    # load dataset
    dataset = CacheDataSet(args, dataset_path=args.dataset_path, disable_pbar=not accelerator.is_main_process, preprocessor=preprocessor)
    test_dataset = [CacheDataSet(args, dataset_path=item, disable_pbar=not accelerator.is_main_process, type="test", preprocessor=preprocessor) for item in args.test_dataset_path]
    dataset_size = len(dataset)
    val_dataset_size = min(int(args.ratio_eval * dataset_size), 10000)
    train_dataset_size = dataset_size - val_dataset_size
    train_dataset, val_dataset = random_split(dataset, [train_dataset_size, val_dataset_size])
    if accelerator.is_main_process:
        print('train_dataset_size', train_dataset_size, 'val_dataset_size', val_dataset_size, 'test_dataset_size', len(test_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True, prefetch_factor=args.prefetch_factor)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=False, prefetch_factor=args.prefetch_factor)
    test_dataloader = [DataLoader(item, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=False, prefetch_factor=args.prefetch_factor) for item in test_dataset]

    net = IDM(model_name=args.model_name, output_dim=14)

    optimizer = AdamW(net.parameters())
    net.train()
    loss_fn = nn.SmoothL1Loss()

    # Setup learning rate scheduler
    if args.lr_scheduler == "cosine":
        warmup_steps = int(0.1 * args.num_iterations)  # 10% of total steps for warmup
        def lr_lambda(step):
            step = step // num_gpus
            eta_min = 1e-9
            if step < warmup_steps:
                # Linear warmup
                return eta_min + float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(max(1, args.num_iterations - warmup_steps))
            return 0.5 * (np.cos(progress * np.pi) + 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if accelerator.is_main_process:
            print(f"Using cosine scheduler with {warmup_steps} warmup steps")
        if accelerator.is_main_process:
            print(f"Using cosine decay scheduler with {warmup_steps} warmup steps")
    else:
        scheduler = None

    if not args.load_from or not os.path.isfile(args.load_from):
        if args.eval_only:
            raise ValueError("Must specify --load_from with a valid model path when using --eval_only")
        start_step = 0
    else:
        try:
            loaded_dict = torch.load(args.load_from)
            net.load_state_dict(loaded_dict["model_state_dict"])
            if not args.eval_only:
                optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                start_step = loaded_dict["step"]
                if scheduler is not None:
                    for _ in range(start_step):
                        scheduler.step()
            if accelerator.is_main_process:
                print(f"Loaded model from {args.load_from}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {args.load_from}: {str(e)}")

    net, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        net, optimizer, train_dataloader, val_dataloader)
    test_dataloader = [accelerator.prepare(dataloader) for dataloader in test_dataloader]
    net.normalize = accelerator.unwrap_model(net).normalize
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    if args.eval_only:
        preprocessor.use_transform = False
        eval(accelerator, net, val_dataloader, loss_fn, 0, args.use_normalization, mode='val', save_dir=save_dir)
        for i in range(len(test_dataloader)):
            eval(accelerator, net, test_dataloader[i], loss_fn, 0, args.use_normalization, mode=f'test{i}', save_dir=save_dir)
        preprocessor.use_transform = args.use_transform
        return

    train_data_generator = get_data_generator(train_dataloader)

    pbar = tqdm(range(start_step, args.num_iterations), disable=not accelerator.is_main_process)
    for step in pbar: 
        images, pos = next(train_data_generator)
        output = net(images, return_mask=True)
        if isinstance(output, tuple):
            output, mask = output
        else:
            mask = None

        # Calculate batch accuracy using denormalized values
        batch_correct = is_close(pos, output)
        batch_accuracy = batch_correct.float().mean().item()

        if args.use_normalization:
            loss = loss_fn(net.normalize(output), net.normalize(pos))
        else:
            loss = loss_fn(output, pos)
        if mask is not None:
            mask_loss = args.mask_weight * mask.mean()
            loss += mask_loss
        else:
            mask_loss = torch.tensor(0.0, device=loss.device)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.2e}", mask_loss=f"{mask_loss.item():.2e}", lr=f"{current_lr:.2e}", batch_acc=f"{batch_accuracy:.4f}")
            if step % 10 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "mask_loss": mask_loss.item(),
                    "learning_rate": current_lr,
                    "batch_accuracy": batch_accuracy
                }, step=step)

        if (step + 1) % args.eval_interval == 0:
            try:
                preprocessor.use_transform = False
                eval(accelerator, net, val_dataloader, loss_fn, step, args.use_normalization, mode='val', save_dir=save_dir)
                for i in range(len(test_dataloader)):
                    eval(accelerator, net, test_dataloader[i], loss_fn, step, args.use_normalization, mode=f'test{i}', save_dir=save_dir)
                preprocessor.use_transform = args.use_transform
            except Exception as e:
                print(f"Error during evaluation at step {step}: {str(e)}")

            save_model(accelerator, net, optimizer, step + 1, os.path.join(save_dir, f"{step + 1}.pt"))
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main(parse_args())
