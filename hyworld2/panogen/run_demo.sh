python pipeline.py \
    --pretrained-model-name-or-path /path/to/HY-Pano-2.0 \
    --subfolder "" \
    --image /path/to/input.png \
    --prompt "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style." \
    --save temp/out_panorama.png

# python pipeline_with_qwen_image.py \
#     --lora-path /path/to/lora \
#     --lora-subfolder "" \
#     --image /path/to/input.png \
#     --prompt "A sunny outdoor scene." \
#     --save temp/out_panorama.png

# HunyuanImage-3 backend (80B MoE, needs >=4x72GB GPUs):
cd "$(dirname "$0")" && CUDA_VISIBLE_DEVICES=1,4,5,6 python pipeline.py --image "../../examples/panogen/abandoned gas station.png"

# Qwen-Image-Edit-2509 + HY-Pano-2.0 LoRA backend (~61GB, single GPU):
cd "$(dirname "$0")" && CUDA_VISIBLE_DEVICES=0 python pipeline_with_qwen_image.py --image "../../examples/panogen/abandoned gas station.png"
