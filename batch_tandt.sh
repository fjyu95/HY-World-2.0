#!/bin/bash
# batch_tandt.sh — 4 T&T/DB scenes on free GPUs, with COLMAP camera priors
#
#   Usage: bash batch_tandt.sh [MAX_FRAMES] [NUM_GPUS]   (default: 32, 4)
#
#   - subsamples each scene to <=MAX_FRAMES frames
#   - injects COLMAP camera params via --prior_cam_path (c2w + K)
#   - no depth maps in dataset -> --prior_depth_path not used
#   - MAX_FRAMES >= 64: drjohnson/playroom additionally get
#     --target_size 800 (their 630px-tall frames OOM at the 952 default)
#
#   Outputs: inference_output_tandt<MAX_FRAMES>/<scene>/<ts>/
#   Logs   : inference_output_tandt<MAX_FRAMES>/logs/<scene>_gpu<N>.log
set -u
cd "$(dirname "$0")"

PY=/opt/miniconda3/envs/hyworld2/bin/python
SRC=examples/worldrecon/tandt_db
MAXF=${1:-32}
NUM=${2:-4}
FREE_MEM_MIB=500

WORK=inference_input_tandt$MAXF
PRIORS=$WORK/priors
OUT=inference_output_tandt$MAXF
mkdir -p "$OUT/logs"

mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -v t=$FREE_MEM_MIB '$2 < t {print $1}' | head -n "$NUM")
if [ ${#GPUS[@]} -eq 0 ]; then echo "no free GPU found"; exit 1; fi
echo "using GPUs: ${GPUS[*]}, MAX_FRAMES=$MAXF"

SCENES=(train truck drjohnson playroom)
declare -A DIR=(
  [train]=$SRC/tandt/train
  [truck]=$SRC/tandt/truck
  [drjohnson]=$SRC/db/drjohnson
  [playroom]=$SRC/db/playroom
)
# >=64 frames: tall db scenes need a smaller inference resolution
declare -A EXTRA=([train]="" [truck]="" [drjohnson]="" [playroom]="")
if [ "$MAXF" -ge 64 ]; then
    EXTRA[drjohnson]="--target_size 800"
    EXTRA[playroom]="--target_size 800"
fi

mem_sampler() {
    while true; do
        nvidia-smi --query-gpu=$1 --query-gpu=memory.used \
            --format=csv,noheader,nounits >> "$2"
        sleep 3
    done
}

# --- stage 1: subsample + convert priors (serial, fast) ---
for scene in "${SCENES[@]}"; do
    src=${DIR[$scene]} dst=$WORK/$scene
    mkdir -p "$dst" "$PRIORS"
    mapfile -t imgs < <(find "$src/images" -maxdepth 1 -type f | sort)
    n=${#imgs[@]}
    if [ "$n" -le "$MAXF" ]; then sel=("${imgs[@]}")
    else
        sel=()
        for i in $(seq 0 $((MAXF-1))); do
            sel+=("${imgs[$(( i * (n - 1) / (MAXF - 1) ))]}")
        done
    fi
    for f in "${sel[@]}"; do cp -n "$f" "$dst/"; done
    "$PY" utils/colmap_to_prior.py "$src/sparse/0" "$dst" "$PRIORS/$scene.json"
    echo "[$scene] $n imgs -> ${#sel[@]} frames, prior -> $PRIORS/$scene.json"
done

# --- stage 2: parallel workers ---
worker() {        # $1=gpu $2=scene
    local gpu=$1 scene=$2
    local log=$OUT/logs/${scene}_gpu${gpu}.log
    local memlog=$OUT/logs/${scene}_gpu${gpu}.mem.log
    : > "$memlog"
    mem_sampler "$gpu" "$memlog" &
    local sampler_pid=$!

    echo "===== [GPU$gpu] $(date '+%F %T') START $scene (maxf=$MAXF ${EXTRA[$scene]}) =====" >> "$log"
    env -u http_proxy -u https_proxy -u all_proxy -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY \
        HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=$gpu "$PY" -m hyworld2.worldrecon.pipeline \
        --input_path "$WORK/$scene" \
        --prior_cam_path "$PRIORS/$scene.json" \
        ${EXTRA[$scene]} \
        --output_path "$OUT" >> "$log" 2>&1
    local rc=$?

    kill "$sampler_pid" 2>/dev/null
    local peak=$(sort -n "$memlog" | tail -1)
    local metrics=$(grep -oE "\[Inference\].*Done in [0-9.]+s|Adaptive resolution.*|\[Memory\] allocated=[0-9.]+GB|[0-9]+ images, shape=torch.Size\([^)]*\)|Prior.*" "$log")
    {
        echo "===== [GPU$gpu] $(date '+%F %T') EXIT=$rc $scene ====="
        echo "----- metrics -----"
        echo "$metrics"
        echo "PEAK_GPU_MEM_MIB=${peak:-NA}"
    } >> "$log"
    [ $rc -ne 0 ] && echo "[GPU$gpu] FAILED($rc) $scene -> $log"
}

i=0
for scene in "${SCENES[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    worker "$gpu" "$scene" &
    i=$((i+1))
done
wait
echo "ALL JOBS FINISHED"
