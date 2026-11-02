import torch, importlib.util
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "device_count:", torch.cuda.device_count())
try:
    import flash_attn
    print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))
    spec = importlib.util.find_spec("flash_attn_2_cuda")
    print("flash_attn_2_cuda .so:", spec.origin if spec else "NOT FOUND")
except Exception as e:
    print("flash_attn import FAILED:", e)
