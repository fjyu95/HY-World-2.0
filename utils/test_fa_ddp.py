# test_fa_ddp.py
import os, torch
from flash_attn.flash_attn_interface import flash_attn_func

rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(rank)
device = f"cuda:{rank}"

B, H, T, D = 1, 8, 128, 64
q = torch.randn(B, T, H, D, device=device, dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
(out.float().mean()).backward()
print(f"[rank{rank}] flash_attn ok on {device}, out={tuple(out.shape)}")
