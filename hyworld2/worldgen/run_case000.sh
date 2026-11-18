#!/usr/bin/env bash
# Full worldgen pipeline example on examples/worldgen/case000.
# Prereqs: conda env `hyworld2`; vLLM Qwen3-VL server up (see vllm_setup.md);
#          models pre-downloaded (README "Pre-download" block).
set -euo pipefail
cd "$(dirname "$0")"

TARGET_PATH=${TARGET_PATH:-../../examples/worldgen/case000}
RESULT_DIR=${RESULT_DIR:-$TARGET_PATH/output}
LLM_ADDR=${LLM_ADDR:-localhost}
LLM_PORT=${LLM_PORT:-8001}
LLM_NAME=${LLM_NAME:-Qwen/Qwen3-VL-8B-Instruct}

# GPUs for multi-GPU stages (vLLM server GPUs are separate)
GPUS=${GPUS:-0}
FIRST_GPU=${GPUS%%,*}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)

run_cmd() {
    echo "CUDA_VISIBLE_DEVICES=$GPUS $*"
    CUDA_VISIBLE_DEVICES=$GPUS "$@"
}

echo "==> Stage 1: Trajectory Planning (single GPU)"
# run_cmd python traj_generate.py --target_path "$TARGET_PATH" \
#     --llm_addr "$LLM_ADDR" --llm_port "$LLM_PORT" --llm_name "$LLM_NAME" \
#     --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm

# 1/2/4卡 97s 72s 72s 
# exit 0

echo "==> Stage 2: Trajectory Rendering ($NPROC GPUs)"
# run_cmd torchrun --nproc_per_node "$NPROC" traj_render.py --target_path "$TARGET_PATH" \
#     --llm_addr "$LLM_ADDR" --llm_port "$LLM_PORT" --llm_name "$LLM_NAME"

# 65s 42s 32s
# exit 0

echo "==> Stage 3: World Expansion / Keyframe Generation ($NPROC GPUs + FSDP)"
# run_cmd torchrun --nproc_per_node "$NPROC" video_gen.py --target_path "$TARGET_PATH" --fsdp

# 1/2/4 OOM/505s/70min
# exit 0

echo "==> Stage 4: GS Training Data ($NPROC GPUs)"
# run_cmd torchrun --nproc_per_node "$NPROC" gen_gs_data.py  --root_path "$TARGET_PATH" --save_normal --split_sky

# 1/2/4 114s/58s/40s
# exit 0

echo "==> Stage 5: 3DGS Training ($NPROC GPUs)"
# Fewer GPUs -> scale steps: x4:2000  x2:4000  x1:8000
# run_cmd python -m world_gs_trainer default \
#     --data_dir "$TARGET_PATH/gs_data" --result_dir "$RESULT_DIR" \
#     --max_steps 1500 --save_steps 1500 --eval_steps 1500 --ply_steps 1500 \
#     --save_ply --convert_to_spz --disable_video  \
#     --use_scale_regularization --antialiased \
#     --depth_loss --normal_loss --sky_depth_from_pcd \
#     --use_mask_gaussian --mask_export_stochastic \
#     --no-mask-export-anchor-protection --use_anchor_protection  \
#     --strategy.refine-start-iter 150 --strategy.refine-stop-iter 750 \
#     --strategy.refine-every 100 --strategy.refine-scale2d-stop-iter 750 \
#     --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1 \
#     --disable_viewer # 单卡建议开，不会Viewer running... Ctrl+C to exit，多卡会自动禁用
#     # --export_mesh 

# 不输出mesh，1/2/4  2/3.5min/15min
# exit 0

# Stage 6 (optional): standalone mesh export from saved ckpts — no retraining needed.
# Use this if Stage 5 ran WITHOUT --export_mesh. Renders RGBD from all cameras,
# fuses via CPU TSDF (GPU VoxelBlockGrid hangs on sm_120 with open3d<=0.18).
# -> writes $RESULT_DIR/ply/fuse_post.ply and fuse_simplified.ply
echo "==> Stage 6: Mesh Export (single GPU, CPU TSDF ~4.5min)"
# CUDA_VISIBLE_DEVICES=$FIRST_GPU python export_mesh.py --gpu_id 0 --data_dir "$TARGET_PATH/gs_data" \
#     --ckpt "$RESULT_DIR/ckpts/ckpt_1499_rank*.pt" \
#     --out_dir "$RESULT_DIR/ply" --voxel_size 0.1

# 实际走的是CPU，4.5min
# TODO: GPU加速版本
# exit 0

# Viewer: visualize the trained 3DGS (blocking until Ctrl+C)
HOST_IP=$(hostname -I | awk '{print $1}')
echo "==> Viewer: after it starts, open http://$HOST_IP:8081        # same LAN"
echo "    or: ssh -L 8081:localhost:8081 <user>@$HOST_IP   # then http://localhost:8081"
CUDA_VISIBLE_DEVICES=$FIRST_GPU python show_gs.py --port 8081 --gpu_id 0 --ckpt "$RESULT_DIR/ckpts/ckpt_1499_rank*.pt"
