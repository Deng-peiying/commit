#!/bin/bash
source activate vidar
DATASET_PATH=./assets/train
TEST_DATASET_PATH=./assets/test
SAVE_DIR=./output/$1
accelerate launch --main_process_port=29348 train_idm.py --learning_rate 5e-4 --use_normalization --use_transform --batch_size 16 --num_iterations 60000 --eval_interval 5000 --num_workers 8 --prefetch_factor 8 --dataset_path $DATASET_PATH --test_dataset_path $TEST_DATASET_PATH --save_dir $SAVE_DIR --wandb_mode offline --model_name mask --mask_weight 3e-3 --run_name "VIDAR" --eval_only
