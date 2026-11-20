#!/usr/bin/env bash
# Full worldgen pipeline example on examples/worldgen/case000.
# Prereqs: conda env `hyworld2`; vLLM Qwen3-VL server up (see vllm_setup.md);
#          models pre-downloaded (README "Pre-download" block).
#
# Timing summary (1/2/4 GPUs; measured WITHOUT navmesh, i.e. no --apply_nav_traj):
#   Stage 1 Trajectory Planning :  97s   / 72s    / 72s
#   Stage 2 Trajectory Rendering:  65s   / 42s    / 32s
#   Stage 3 World Expansion     :  OOM   / 505s   / 70min  (FSDP data-parallel, may be slower)
#   Stage 4 GS Training Data    :  114s  / 58s    / 40s
#   Stage 5 3DGS Training       :  2min  / 3.5min / 15min  (no mesh export)
#   Stage 6 Mesh Export         :  ~4.5min (CPU TSDF, GPU-independent)
#   Total                       :  ~11min (excl. Stage 3, which OOMs) / ~19min / ~92min (~1.5h)

# Timing summary WITH navmesh (after navmesh fix, 1/2/4 GPUs):
#   Stage 1 Trajectory Planning :  ~16min  / ~16min  / ~16min  (end-to-end; timer-tracked parts ~281.7s, rest is model loading + vLLM)
#   Stage 2 Trajectory Rendering:  580s    / 360s    / 220s
#   Stage 3 World Expansion     :  OOM     / 1800s   / 3h05min
#   Stage 4 GS Training Data    :  160s    / 75s     / 48s
#   Stage 5 3DGS Training       :  2min    / 3.5min  / 13.5min (script pins $FIRST_GPU; multi-GPU values are historical)
#   Stage 6 Mesh Export         :  216s    / 212s     / 200s    (CPU TSDF, GPU-independent)
#   Total                       :  ~34min (excl. Stage 3, which OOMs) / ~57min / ~3.7h
#   Note: Stage 3/5 slower with more GPUs is expected — PRO 5000 has no NVLink,
#         DDP/FSDP sync goes over PCIe so communication cost grows with GPU count.

set -euo pipefail
cd "$(dirname "$0")"

TARGET_PATH=${TARGET_PATH:-../../examples/worldgen/case000}
RESULT_DIR=${RESULT_DIR:-$TARGET_PATH/output}
LLM_ADDR=${LLM_ADDR:-localhost}
LLM_PORT=${LLM_PORT:-8001}
LLM_NAME=${LLM_NAME:-Qwen/Qwen3-VL-8B-Instruct}

# GPUs for multi-GPU stages (vLLM server GPUs are separate)
GPUS=${GPUS:-4,5}
FIRST_GPU=${GPUS%%,*}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# run_cmd [gpu_list] <cmd...>: optional first arg overrides CUDA_VISIBLE_DEVICES,
# defaults to $GPUS. Pass "$FIRST_GPU" for single-GPU stages.
run_cmd() {
    local vis=$GPUS
    if [[ $1 =~ ^[0-9,]+$ ]]; then vis=$1; shift; fi
    echo "CUDA_VISIBLE_DEVICES=$vis $*"
    CUDA_VISIBLE_DEVICES=$vis "$@"
}

echo "==> Stage 1: Trajectory Planning ($NPROC GPU)"
run_cmd python traj_generate.py --target_path "$TARGET_PATH" \
    --llm_addr "$LLM_ADDR" --llm_port "$LLM_PORT" --llm_name "$LLM_NAME" \
    --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm

# exit 0

echo "==> Stage 2: Trajectory Rendering ($NPROC GPUs)"
run_cmd torchrun --nproc_per_node "$NPROC" traj_render.py --target_path "$TARGET_PATH" \
    --llm_addr "$LLM_ADDR" --llm_port "$LLM_PORT" --llm_name "$LLM_NAME"

# exit 0

echo "==> Stage 3: World Expansion / Keyframe Generation ($NPROC GPUs + FSDP)"
# 必须 --fsdp 数据并行，否则单卡OOM
run_cmd torchrun --nproc_per_node "$NPROC" video_gen.py --target_path "$TARGET_PATH" --fsdp 

# exit 0

echo "==> Stage 4: GS Training Data ($NPROC GPUs)"
run_cmd torchrun --nproc_per_node "$NPROC" gen_gs_data.py  --root_path "$TARGET_PATH" --save_normal --split_sky

# exit 0

echo "==> Stage 5: 3DGS Training ($NPROC GPUs)"
run_cmd "$FIRST_GPU" python -m world_gs_trainer default \
    --data_dir "$TARGET_PATH/gs_data" --result_dir "$RESULT_DIR" \
    --max_steps 1500 --save_steps 1500 --eval_steps 1500 --ply_steps 1500 \
    --save_ply --convert_to_spz --disable_video  \
    --use_scale_regularization --antialiased \
    --depth_loss --normal_loss --sky_depth_from_pcd \
    --use_mask_gaussian --mask_export_stochastic \
    --no-mask-export-anchor-protection --use_anchor_protection  \
    --strategy.refine-start-iter 150 --strategy.refine-stop-iter 750 \
    --strategy.refine-every 100 --strategy.refine-scale2d-stop-iter 750 \
    --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1 \
    --disable_viewer # 单卡建议开，不会Viewer running... Ctrl+C to exit，多卡会自动禁用
    # --export_mesh 

# exit 0

# Stage 6 (optional): standalone mesh export from saved ckpts — no retraining needed.
# Use this if Stage 5 ran WITHOUT --export_mesh. Renders RGBD from all cameras,
# fuses via CPU TSDF (GPU VoxelBlockGrid hangs on sm_120 with open3d<=0.18).
# -> writes $RESULT_DIR/ply/fuse_post.ply and fuse_simplified.ply
echo "==> Stage 6: Mesh Export (single GPU, CPU TSDF ~4.5min)"
run_cmd "$GPUS" python export_mesh.py --gpu_id 0 --data_dir "$TARGET_PATH/gs_data" \
    --ckpt "$RESULT_DIR/ckpts/ckpt_1499_rank*.pt" \
    --out_dir "$RESULT_DIR/ply" --voxel_size 0.1

# TODO: GPU加速版本
# exit 0

# Viewer: visualize the trained 3DGS (blocking until Ctrl+C)
HOST_IP=$(hostname -I | awk '{print $1}')
echo "==> Stage 7 Viewer: after it starts, open http://$HOST_IP:8081        # same LAN"
echo "    or: ssh -L 8081:localhost:8081 <user>@$HOST_IP   # then http://localhost:8081"
run_cmd "$FIRST_GPU" python show_gs.py --port 8081 --gpu_id 0 --ckpt "$RESULT_DIR/ckpts/ckpt_1499_rank*.pt"
