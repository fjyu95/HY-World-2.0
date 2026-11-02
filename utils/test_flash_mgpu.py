# save as test_flash_mgpu.py
import torch
import flash_attn
from flash_attn import flash_attn_func

print(f"flash_attn {flash_attn.__version__}  GPUs={torch.cuda.device_count()}")

# 构造多卡输入 (batch=2, 2GPU)
B, H, N, D = 2, 8, 1024, 64
q = torch.randn(B, N, H, D, dtype=torch.float16, device='cuda:0')
k = torch.randn(B, N, H, D, dtype=torch.float16, device='cuda:0')
v = torch.randn(B, N, H, D, dtype=torch.float16, device='cuda:0')

# 可选：模型并行/数据并行前向
with torch.cuda.amp.autocast(dtype=torch.float16):
    out = flash_attn_func(q, k, v, causal=True)
print("✅  multi-GPU forward passed", out.shape)
