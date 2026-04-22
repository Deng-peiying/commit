"""快速验证模型是否真的依赖视觉信息。

模型预测绝对 pos_{t+1}（不是增量 delta）。

测试方法：
1. 正常输入 (img_t, img_next)        → pred_normal  应该 ≈ pos_next
2. 两帧相同 (img_t, img_t)           → pred_same    feat_diff=0 → 应该 ≈ pos_t
3. 错误真实图片 (img_t, other_next)  → pred_wrong   不相关的真实帧 → 应该和 normal 差别大

如果模型真的在用视觉：
  - pred_same ≈ pos_t（没有运动 → 预测不变）
  - pred_wrong 和 pred_normal 差别大（错误的真实视觉 → 错误预测）
  - pred_normal ≈ pos_next

如果模型在抄近路（只用 pos_t）：
  - 三种情况输出几乎一样（都 ≈ 某个和 pos_t 相关的值）
"""

import os
import sys
import glob
import torch
import argparse
from idm.idm import IDM, OUTPUT_DIM
from idm.cache_dataset import CacheDataSet
from idm.preprocessor import DinoPreprocessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/assets/train")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    # 如果传入的是目录，自动找最新的 checkpoint
    if os.path.isdir(args.checkpoint):
        ckpts = sorted(glob.glob(os.path.join(args.checkpoint, "*.pt")))
        if not ckpts:
            print(f"No .pt files found in {args.checkpoint}")
            sys.exit(1)
        args.checkpoint = ckpts[-1]
        print(f"Auto-selected checkpoint: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    net = IDM(model_name="mask", output_dim=OUTPUT_DIM)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device).eval()

    # 加载数据（不做增强）
    dummy_args = argparse.Namespace(use_transform=False)
    preprocessor = DinoPreprocessor(dummy_args)
    dataset = CacheDataSet(dummy_args, dataset_path=args.dataset_path,
                           disable_pbar=True, preprocessor=preprocessor)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Testing {args.num_samples} samples...\n")
    print(f"{'idx':>4}  {'motion':>7}  {'baseline':>9}  {'normal err':>10}  {'same→pos_t':>11}  {'wrong≠norm':>11}  "
          f"{'beat?':>6}  {'same≈t?':>8}  {'wrong≠n?':>9}")
    print("-" * 100)

    import random

    # 只选运动帧测试（|pos_next - pos_t| 均值 > 0.03）
    motion_indices = []
    for i in range(len(dataset)):
        _, _, _, _, pt, pn = dataset[i]
        if (pn - pt).abs().mean().item() > 0.03:
            motion_indices.append(i)
        if len(motion_indices) >= args.num_samples * 5:  # 搜索够多就停
            break

    random.shuffle(motion_indices)
    test_indices = motion_indices[:args.num_samples]

    with torch.no_grad():
        for i in test_indices:
            img_t, dep_t, img_next, dep_next, pos_t, pos_next = dataset[i]

            # 从数据集随机选一个不同样本的 img_next 作为"错误的真实图片"
            j = i
            while j == i:
                j = random.randint(0, len(dataset) - 1)
            _, _, other_img_next, other_dep_next, _, _ = dataset[j]

            # 移到 GPU，加 batch 维度
            img_t    = img_t.unsqueeze(0).to(device)
            dep_t    = dep_t.unsqueeze(0).to(device)
            img_next = img_next.unsqueeze(0).to(device)
            dep_next = dep_next.unsqueeze(0).to(device)
            pos_t    = pos_t.unsqueeze(0).to(device)
            pos_next = pos_next.unsqueeze(0).to(device)
            other_img_next = other_img_next.unsqueeze(0).to(device)
            other_dep_next = other_dep_next.unsqueeze(0).to(device)

            # 1) 正常输入 → 应该 ≈ pos_next
            pred_normal = net(img_t, dep_t, img_next, dep_next, pos_t)

            # 2) 两帧相同 → feat_diff=0 → 应该 ≈ pos_t（没有运动）
            pred_same = net(img_t, dep_t, img_t, dep_t, pos_t)

            # 3) 用另一个样本的真实图片替换 img_next → 不相关的运动信号
            pred_wrong = net(img_t, dep_t, other_img_next, other_dep_next, pos_t)

            # 指标
            motion       = (pos_next - pos_t).abs().mean().item()          # 实际运动量
            baseline_err = motion                                           # baseline: 直接输出 pos_t 的误差 = motion
            normal_err   = (pred_normal - pos_next).abs().mean().item()    # 正常预测误差
            same_to_t    = (pred_same - pos_t).abs().mean().item()         # 两帧相同 → 离 pos_t 多远
            wrong_vs_norm = (pred_wrong - pred_normal).abs().mean().item() # 错误真实图 vs 正常的差异

            beat_ok  = "✅" if normal_err < baseline_err * 0.5 else "❌"   # 模型误差 < baseline 的 50%
            same_ok  = "✅" if same_to_t < 0.05 else "❌"
            wrong_ok = "✅" if wrong_vs_norm > 0.02 else "❌"

            print(f"{i:4d}  {motion:7.4f}  {baseline_err:9.4f}  {normal_err:10.4f}  {same_to_t:11.4f}  {wrong_vs_norm:11.4f}  "
                  f"{beat_ok:>6}  {same_ok:>8}  {wrong_ok:>9}")

    print("\n判读标准：")
    print("  beat?    ✅ = 模型误差 < baseline(原地不动)的50% → 模型不是原地踏步，真的在预测运动")
    print("  beat?    ❌ = 模型误差 ≥ baseline的50% → 模型可能在原地踏步")
    print("  same≈t?  ✅ = 两帧相同时输出≈pos_t → 模型依赖视觉差异（feat_diff=0 → 不动）")
    print("  same≈t?  ❌ = 两帧相同时输出远离pos_t → 模型可能在抄近路")
    print("  wrong≠n? ✅ = 换了不相关的真实图片，输出明显不同 → 模型在用视觉语义")
    print("  wrong≠n? ❌ = 换了不相关的真实图片，输出几乎不变 → 模型忽略了视觉")

    # ================================================================
    # 严格测试：固定 pos_t，只换视觉 → 输出是否不同？
    # 用样本 A 的 pos_t，分别配 A 的视觉和 B 的视觉
    # 如果模型只学了 "pos_t → 运动方向"，两次输出应该一样
    # ================================================================
    print("\n" + "=" * 80)
    print("严格测试：固定同一个 pos_t，只换视觉输入 → 输出是否不同？")
    print("如果模型只学了 'pos_t → 通常往哪动'，换视觉不应该改变输出")
    print("=" * 80)

    pairs_found = 0
    print(f"\n{'pair':>5}  {'idx_A':>6}  {'idx_B':>6}  "
          f"{'pred_A err':>10}  {'pred_swapB err':>12}  {'A≠swapB':>8}  {'视觉有用?':>10}")
    print("-" * 75)

    # 随机选运动帧配对
    random.shuffle(motion_indices)

    with torch.no_grad():
        for p in range(0, len(motion_indices) - 1, 2):
            if pairs_found >= 10:
                break
            idx_a = motion_indices[p]
            idx_b = motion_indices[p + 1]

            img_t_a, dep_t_a, img_next_a, dep_next_a, pos_t_a, pos_next_a = dataset[idx_a]
            img_t_b, dep_t_b, img_next_b, dep_next_b, pos_t_b, pos_next_b = dataset[idx_b]

            # 移到 GPU
            pos_t_a_gpu = pos_t_a.unsqueeze(0).to(device)
            pos_next_a_gpu = pos_next_a.unsqueeze(0).to(device)

            # pred_A: 样本 A 的 pos_t + 样本 A 的视觉（正常）
            pred_a = net(
                img_t_a.unsqueeze(0).to(device), dep_t_a.unsqueeze(0).to(device),
                img_next_a.unsqueeze(0).to(device), dep_next_a.unsqueeze(0).to(device),
                pos_t_a_gpu)

            # pred_swapB: 样本 A 的 pos_t + 样本 B 的视觉（只换了图片）
            pred_swap = net(
                img_t_b.unsqueeze(0).to(device), dep_t_b.unsqueeze(0).to(device),
                img_next_b.unsqueeze(0).to(device), dep_next_b.unsqueeze(0).to(device),
                pos_t_a_gpu)  # ← 关键：pos_t 用的是 A 的！

            err_a    = (pred_a - pos_next_a_gpu).abs().mean().item()
            err_swap = (pred_swap - pos_next_a_gpu).abs().mean().item()
            pred_diff = (pred_a - pred_swap).abs().mean().item()

            # pos_t 完全一样，如果输出不同，100% 是视觉驱动的
            visual_ok = "✅" if pred_diff > 0.02 else "❌"

            print(f"{pairs_found:5d}  {idx_a:6d}  {idx_b:6d}  "
                  f"{err_a:10.4f}  {err_swap:12.4f}  {pred_diff:8.4f}  {visual_ok:>10}")
            pairs_found += 1

    print("\n判读：")
    print("  pred_A err    = 正常预测误差（A的pos_t + A的视觉）")
    print("  pred_swapB err= 换视觉后的误差（A的pos_t + B的视觉）→ 应该变大")
    print("  A≠swapB       = 两次预测的差异（pos_t 完全相同，差异 100% 来自视觉）")
    print("  视觉有用? ✅  = 差异 > 0.02 → 视觉确实在驱动预测")
    print("  视觉有用? ❌  = 差异 < 0.02 → 模型可能只看 pos_t")


if __name__ == "__main__":
    main()
