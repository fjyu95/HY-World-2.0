# vLLM Server Setup (Qwen3-VL) — hyworld2-vllm

The worldgen pipeline (stages 1–2, `traj_generate.py` / `traj_render.py`) calls a
Qwen3-VL model over the OpenAI-compatible HTTP API (`--llm_addr`, `--llm_port`,
`--llm_name`). This doc covers how to run vLLM server on machine
(8× RTX PRO 5000 Blackwell, `sm_120`, CUDA 12.8) in a **dedicated conda env** so
nothing else gets polluted.

## Version chain (read first)

| torch | vllm | Qwen3-VL | Verdict |
|---|---|---|---|
| 2.7.1+cu128 | ≤0.10.1.1 | `TransformersForMultimodalLM` fallback | ❌ **cannot serve Qwen3-VL at all** |
| **2.8.0+cu128** | **0.11.0** | **Native `Qwen3VLForConditionalGeneration`, TP works** | ✅ use this |

### Why vllm 0.10.x fails for Qwen3-VL (both single- and multi-GPU)

vllm 0.10.x has no native `Qwen3VLForConditionalGeneration` implementation;
the model resolves to the generic `TransformersForMultimodalLM` fallback
backend, which is broken for this architecture in two ways:

- **TP=2+**: `ValueError: Qwen3VLModel does not support tensor parallel yet!`
  — the fallback backend cannot shard this model.
- **TP=1**: weights load fine, then `profile_run()` dies with
  `AssertionError: Expected number of multimodal embeddings to match number
  of input items: 1, but got len(mm_embeddings)=2` — the fallback's
  `get_multimodal_embeddings` returns a mismatched embedding count. So even
  a single GPU cannot serve it; there is **no usable 0.10.x path**, the
  correct answer is upgrading to 0.11.0 (which requires torch 2.8.0).

- **transformers must be 4.x** — vllm 0.11.0 declares `transformers>=4.55.2` with
  no upper bound, so pip happily installs 5.x which removed
  `all_special_tokens_extended` → `AttributeError` at tokenizer init.
  Pin `transformers==4.57.1`.
- **flash-attn is NOT needed** for vllm (kernels are vendored). Our
  `flash_attn 2.8.3` is only for WorldMirror.
- **`--trust-remote-code` is harmless but unnecessary** — Qwen3-VL is a
  built-in transformers architecture; vllm logs that the flag is ignored.
  We keep it in the launch command only for consistency with the upstream
  README.
- vllm 0.10.x workers show as `spawn_main --multiprocessing-fork_TP*`;
  0.11.x as `VLLM::Worker_TP*`. Same mechanism, different naming.

## Build the env

```bash
conda create -n hyworld2-vllm python=3.10 -y
conda activate hyworld2-vllm
```

### Step 1 — torch 2.8.0+cu128 (manual wheel = fastest)

The pip index is slow for the ~850MB torch wheel; download it directly:

```bash
mkdir -p ~/pkg && cd ~/pkg
wget -c https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl

# Install local wheel — but let pip resolve nvidia-* deps from the SAME
# fast CDN (they are ~1.5GB combined: cudnn/cublas/nccl/...):
pip install ./torch-2.8.0+cu128-*.whl --index-url https://download.pytorch.org/whl/cu128
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

`download.pytorch.org/whl/cu128` mirrors all `nvidia-*-cu12` wheels torch
needs, so the whole dependency chain uses the fast CDN.

### Step 2 — torchvision / torchaudio (matching pair)

```bash
pip install torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 3 — vllm 0.11.0 (+ pinned transformers)

```bash
# Grab the wheel URL from PyPI metadata, then wget (proxy ON, ~23MB/s):
URL=$(curl -s https://pypi.org/pypi/vllm/0.11.0/json | \
  python3 -c "import json,sys; print([f['url'] for f in json.load(sys.stdin)['urls'] if 'manylinux' in f['filename'] and f['filename'].endswith('.whl')][0])")
wget -c "$URL" -P ~/pkg/

# Install local wheel; small deps via a mirror; pin transformers<5:
pip install ~/pkg/vllm-0.11.0-*.whl "transformers==4.57.1" \
    -i https://mirrors.aliyun.com/pypi/simple/

pip install qwen-vl-utils openai
```

### Verify

```bash
python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.get_arch_list())"
# expect: 2.8.0+cu128  0.11.0  ['sm_70', ..., 'sm_120']
```

## Download-speed cheat sheet

- **big CUDA wheels** (torch, nvidia-*) → `download.pytorch.org/whl/cu128`
  (~65MB/s). Tuna mirror drops to ~0.4MB/s on these files.
- **vllm wheel** → `files.pythonhosted.org` via proxy (~23MB/s); get URL from
  `pypi.org/pypi/vllm/<ver>/json`.
- **small pure-python deps** → aliyun/tuna mirrors are fine.
- `wget <known-URL>` beats `pip install` for large files — pip pays per-request
  resolver overhead and pulls one file at a time.

## Launch the server

```bash
conda activate hyworld2-vllm
CUDA_VISIBLE_DEVICES=5,6 vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct \
    --port 8001 --host 0.0.0.0 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.80
```

Ready check:

```bash
curl -s http://localhost:8001/v1/models | head -20
```

## Use from worldgen

Client runs in `hyworld2` (server env is irrelevant over HTTP):

```bash
conda activate hyworld2
python traj_generate.py --target_path ../../examples/worldgen/case000 \
    --llm_addr localhost --llm_port 8001 \
    --llm_name Qwen/Qwen3-VL-8B-Instruct \
    --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Qwen3VLModel does not support tensor parallel yet!` | vllm 0.10.x lacks native Qwen3-VL → transformers fallback | vllm 0.11.0 + torch 2.8.0 (this doc) |
| `Expected number of multimodal embeddings to match number of input items: 1, but got len(mm_embeddings)=2` | vllm 0.10.1.1 fallback is broken for Qwen3-VL even at TP=1 — **0.10.x cannot serve Qwen3-VL at all** | same as above |
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5.x installed | `pip install transformers==4.57.1` |
| pip resolving torch deps at ~100KB/s | nvidia-* wheels fetched from tuna | `--index-url https://download.pytorch.org/whl/cu128` |
| `ReadTimeoutError pypi.nvidia.com` | nvidia dep index unreachable w/o proxy | `clashctl on` or use pytorch index which mirrors them |
| torch replaced by cu126 build | default PyPI wheel isn't cu128 | always install torch via cu128 index/local wheel |
