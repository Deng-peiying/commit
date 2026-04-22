from huggingface_hub import snapshot_download
import os

# 设置镜像站（针对国内网络加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

repo_id = "TianxingChen/RoboTwin2.0"
local_dir = "./RoboTwin_Franka_500"

print("开始下载 Franka Randomized 500 系列数据...")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    # 只包含所有任务文件夹下的 franka_randomized_500.zip
    allow_patterns=["*/franka_randomized_500.zip"],
    # 忽略其他不需要的大文件
    ignore_patterns=["*.gif", "*.mp4", "*/aloha_*", "*/arx_*", "*/gr1_*"],
    resume_download=True
)

print(f"下载完成！文件保存在: {os.path.abspath(local_dir)}")