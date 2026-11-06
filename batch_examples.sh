#!/bin/bash
# batch_examples.sh — run the 16 bundled worldrecon cases on up to 4 free GPUs
#
#   Usage: bash batch_examples.sh [NUM_GPUS]    (default: 4)
#
#   Free GPUs are auto-detected (memory.used < 500 MiB). Cases are
#   distributed round-robin in waves of NUM_GPUS.
#
#   Outputs: inference_output/<case>/<timestamp>/
#   Logs   : inference_output/logs/<case>_gpu<N>.log (+ .mem.log)
set -u
cd "$(dirname "$0")"

PY=/opt/miniconda3/envs/hyworld2/bin/python
OUT=inference_output
NUM=${1:-4}
FREE_MEM_MIB=500

# auto-detect free GPUs (up to NUM)
mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -v t=$FREE_MEM_MIB '$2 < t {print $1}' | head -n "$NUM")
if [ ${#GPUS[@]} -eq 0 ]; then echo "no free GPU found"; exit 1; fi
echo "using GPUs: ${GPUS[*]}"

JOBS=(
  "examples/worldrecon/realistic/Messy_Room"          # 32f cluttered room
  "examples/worldrecon/realistic/Dining_Table"        # 4f  dining
  "examples/worldrecon/stylistic/Cottage_Autumn"      # 2f  stylized cottage
  "examples/worldrecon/realistic/Office/Office.jpg"   # 1f  office
  "examples/worldrecon/realistic/Landmark"            # 32f monument
  "examples/worldrecon/realistic/Desk"                # 2f  tabletop
  "examples/worldrecon/stylistic/Palace"              # 3f  stylized palace
  "examples/worldrecon/realistic/Flower/image_0001.jpg" # 1f flower
  "examples/worldrecon/realistic/Valley"              # 11f mountain valley
  "examples/worldrecon/realistic/Room_Cat"            # 8f  cat room
  "examples/worldrecon/stylistic/A_MARCEAU"           # 2f  stylized scene
  "examples/worldrecon/stylistic/Cat_Girl/Cat_Girl.jpg" # 1f stylized character
  "examples/worldrecon/stylistic/A_Stylized_Kitchen"  # 4f  stylized kitchen
  "examples/worldrecon/realistic/Archway_Tunnel"      # 2f  corridor
  "examples/worldrecon/stylistic/Panda_Wild_West/panda_orange_cat_wildwest.jpeg" # 1f
  "examples/worldrecon/realistic/Park/image_0001.jpg" # 1f  park
)

mem_sampler() {
    while true; do
        nvidia-smi --query-gpu=$1 --query-gpu=memory.used \
            --format=csv,noheader,nounits >> "$2"
        sleep 3
    done
}

worker() {        # $1=gpu $2=input
    local gpu=$1 input=$2 tag
    if [ -d "$input" ]; then tag=$(basename "$input")
    else tag="$(basename "$(dirname "$input")")_$(basename "${input%.*}")"; fi
    local log=$OUT/logs/${tag}_gpu${gpu}.log
    local memlog=$OUT/logs/${tag}_gpu${gpu}.mem.log
    mkdir -p "$OUT/logs"
    : > "$memlog"
    mem_sampler "$gpu" "$memlog" &
    local sampler_pid=$!

    echo "===== [GPU$gpu] $(date '+%F %T') START $input =====" >> "$log"
    env -u http_proxy -u https_proxy -u all_proxy -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY \
        HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=$gpu "$PY" -m hyworld2.worldrecon.pipeline \
        --input_path "$input" --output_path "$OUT" >> "$log" 2>&1
    local rc=$?

    kill "$sampler_pid" 2>/dev/null
    local peak=$(sort -n "$memlog" | tail -1)
    local metrics=$(grep -oE "\[Inference\].*Done in [0-9.]+s|Adaptive resolution.*|\[Memory\] allocated=[0-9.]+GB|[0-9]+ images, shape=torch.Size\([^)]*\)" "$log")
    {
        echo "===== [GPU$gpu] $(date '+%F %T') EXIT=$rc $input ====="
        echo "----- metrics -----"
        echo "$metrics"
        echo "PEAK_GPU_MEM_MIB=${peak:-NA}"
    } >> "$log"
    [ $rc -ne 0 ] && echo "[GPU$gpu] FAILED($rc) $input -> $log"
}

i=0
for input in "${JOBS[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    worker "$gpu" "$input" &
    i=$((i+1))
    if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
echo "ALL JOBS FINISHED"
