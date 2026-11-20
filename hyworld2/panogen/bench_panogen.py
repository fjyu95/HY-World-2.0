#!/usr/bin/env python3
"""
Batch benchmark for HY-Pano-2.0: N random images x 2 backends (qwen -> hunyuan).

Run inside the `hyworld2-pano` conda env:
    conda activate hyworld2-pano
    python bench_panogen.py
    python bench_panogen.py --n 3 --gpus-qwen 5 --gpus-hunyuan 2,3,4,5
"""

import argparse
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_IMG_DIR = THIS_DIR / "../../examples/panogen"

BACKENDS = [
    # (name, script, gpus_env_default)
    ("qwen", "pipeline_with_qwen_image.py", "7"),
    ("hunyuan", "pipeline.py", "3,4,5,6"),  # 80B MoE needs >=4 GPUs
]


def parse_args():
    p = argparse.ArgumentParser("Batch benchmark: qwen + hunyuan pano backends")
    p.add_argument("--img-dir", type=Path,
                   default=Path(os.environ.get("IMG_DIR", DEFAULT_IMG_DIR)),
                   help="Image pool dir")
    p.add_argument("--out-dir", type=Path,
                   default=Path(os.environ.get("OUT_DIR", DEFAULT_IMG_DIR / "bench_results")),
                   help="Output dir (default: <img_dir>/bench_results)")
    p.add_argument("-n", "--n-images", type=int,
                   default=int(os.environ.get("N_IMAGES", 5)),
                   help="Number of random images to test")
    p.add_argument("--gpus-qwen", type=str,
                   default=os.environ.get("GPUS_QWEN", BACKENDS[0][2]),
                   help="CUDA_VISIBLE_DEVICES for Qwen-Image-Edit backend")
    p.add_argument("--gpus-hunyuan", type=str,
                   default=os.environ.get("GPUS_HUNYUAN", BACKENDS[1][2]),
                   help="CUDA_VISIBLE_DEVICES for HunyuanImage-3 backend")
    p.add_argument("--seed", type=int, default=42, help="Generation seed")
    p.add_argument("--sel-seed", type=int, default=None,
                   help="Seed for image selection (default: truly random)")
    return p.parse_args()


def run_one(backend, script, img, gpus, save, seed, log):
    print(f"==> [{backend}] {img}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    proc = subprocess.Popen(
        [sys.executable, script,
         "--image", str(img), "--seed", str(seed), "--save", str(save)],
        cwd=THIS_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    rc = proc.wait()
    status = "done" if rc == 0 else f"FAILED rc={rc}"
    print(f"    [{backend}] {img.stem} {status}")
    log.write(f"    [{backend}] {img.stem} {status}\n")
    return rc == 0


def main():
    args = parse_args()
    img_dir = args.img_dir.resolve()

    all_imgs = sorted(p for p in img_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not all_imgs:
        sys.exit(f"No images found in {img_dir}")
    rng = random.Random(args.sel_seed)
    imgs = rng.sample(all_imgs, min(args.n_images, len(all_imgs)))

    out_dir = args.out_dir
    for name, _, _ in BACKENDS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"bench_{datetime.now():%Y%m%d_%H%M%S}.log"

    print(f"Selected {len(imgs)} images:")
    for p in imgs:
        print(f"  - {p}")
    print(f"Full log: {log_path}")

    gpu_map = {"qwen": args.gpus_qwen, "hunyuan": args.gpus_hunyuan}
    results = {}
    with open(log_path, "w") as log:
        for bi, (name, script, _) in enumerate(BACKENDS, 1):
            print(f"\n==> Backend {bi}/{len(BACKENDS)}: {name} "
                  f"({script}, GPUs={gpu_map[name]})")
            for img in imgs:
                save = out_dir / name / f"{img.stem}_panorama.png"
                results[(name, img.stem)] = run_one(
                    name, script, img, gpu_map[name], save, args.seed, log)

    # Summary: per-image status + [Timing] lines from the log
    print("\n" + "=" * 60)
    print("Per-image results:")
    for (backend, stem), ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {backend}/{stem}")
    print("\n[Timing] lines:")
    timing_re = re.compile(r"\[Timing\] (Inference|Total).*")
    with open(log_path) as f:
        for line in f:
            m = timing_re.search(line)
            if m:
                print(f"  {m.group(0)}")
    print(f"\nBenchmark finished. Outputs in {out_dir}, log: {log_path}")


if __name__ == "__main__":
    main()
