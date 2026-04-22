#!/bin/bash
# Usage: bash scripts/idm/eval.sh <checkpoint_path>
# Example: bash scripts/idm/eval.sh output/dual_frame_no_mask_v1/90000.pt

CHECKPOINT=${1:?请指定 checkpoint 路径，例如: output/dual_frame_no_mask_v1/90000.pt}
DATASET_PATH=data/assets/train
TEST_DATASET_PATH=data/assets/test
SAVE_DIR=output/eval_results

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/user/app/miniconda3/envs/vidar/bin/python train_idm.py \
  --load_from $CHECKPOINT \
  --dataset_path $DATASET_PATH \
  --test_dataset_path $TEST_DATASET_PATH \
  --save_dir $SAVE_DIR \
  --model_name mask \
  --batch_size 8 \
  --eval_batch_size 8 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --wandb_mode offline \
  --run_name eval \
  --eval_only
