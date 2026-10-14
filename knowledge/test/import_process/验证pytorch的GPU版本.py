import torch

# 1. 打印 PyTorch 版本，确认是 GPU 版
print(f"PyTorch版本: {torch.__version__}")
# 期望输出类似：2.0.0+cu118（末尾带有 +cuXXX 字样）

# 2. 打印 PyTorch 内置的 CUDA 运行时版本
print(f"PyTorch CUDA版本: {torch.version.cuda}")
# 期望输出类似：11.8

# 3. 检查 CUDA 是否可用（最关键的一步！）
print(f"CUDA是否可用: {torch.cuda.is_available()}")
# 期望输出：True

# 4. 如果 CUDA 可用，查看显卡数量和信息
if torch.cuda.is_available():
    print(f"检测到GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  显卡 {i}: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("警告：CUDA不可用，PyTorch将在CPU上运行。")