import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 名称: {torch.cuda.get_device_name()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 实际跑一个 tensor 到 GPU 上验证
x = torch.randn(3, 3).cuda()
print(f"Tensor 所在设备: {x.device}")
print("✅ GPU 安装成功！")
