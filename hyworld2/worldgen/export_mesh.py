"""Standalone mesh export for a trained world_gs_trainer checkpoint.

Replicates the --export_mesh tail of world_gs_trainer.py without retraining:
load merged ckpts -> render RGB+ED for every camera -> TSDF fuse -> post-process.

Usage:
    CUDA_VISIBLE_DEVICES=0 python export_mesh.py \
        --data_dir ../../examples/worldgen/case000/gs_data \
        --ckpt "../../examples/worldgen/case000/output/ckpts/ckpt_1499_rank*.pt" \
        --out_dir ../../examples/worldgen/case000/output/ply
"""
import argparse
import json
import math
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import tqdm


def load_splats(ckpt_pattern, device: torch.device):
    patterns = [ckpt_pattern] if isinstance(ckpt_pattern, str) else ckpt_pattern
    ckpt_paths = sorted({path for pattern in patterns for path in glob(pattern)})
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints matched: {ckpt_pattern}")
    chunks = {"means": [], "quats": [], "scales": [], "opacities": [], "sh0": [], "shN": []}
    for p in tqdm.tqdm(ckpt_paths, desc="Loading checkpoints..."):
        ckpt = torch.load(p, map_location=device, weights_only=False)["splats"]
        chunks["means"].append(ckpt["means"])
        chunks["quats"].append(F.normalize(ckpt["quats"], p=2, dim=-1))
        chunks["scales"].append(torch.exp(ckpt["scales"]))
        chunks["opacities"].append(torch.sigmoid(ckpt["opacities"]))
        chunks["sh0"].append(ckpt["sh0"])
        chunks["shN"].append(ckpt["shN"])
    splats = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    print("Number of Gaussians:", splats["means"].shape[0])
    return splats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="gs_data dir (images/ + cameras.json + points.ply)")
    parser.add_argument("--ckpt", type=str, nargs="+", required=True,
                        help="checkpoint paths or globs from the same training run; avoid stale ranks from earlier runs")
    parser.add_argument("--out_dir", type=str, required=True, help="ply output dir")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--voxel_size", type=float, default=0.05)   # same default as cfg
    parser.add_argument("--sdf_trunc", type=float, default=-1.0)    # <0 -> 4*voxel_size
    parser.add_argument("--depth_trunc", type=float, default=5.0)   # same as cfg_depth_trunc
    parser.add_argument("--downsample_perc", type=float, default=0.1)
    parser.add_argument("--num_cluster", type=int, default=1)
    parser.add_argument("--gpu_tsdf", action="store_true",
                        help="use GPU VoxelBlockGrid TSDF (hangs on sm_120/Blackwell with open3d<=0.18)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Same Parser normalization as training (normalize=True, factor=1).
    from gs.opencv import Parser
    parser_obj = Parser(
        data_dir=args.data_dir,
        factor=1,
        normalize=True,
        detect_anchor_candidates=False,
        world_rank=0,
        world_size=1,
        local_rank=args.gpu_id,
    )
    camtoworlds = parser_obj.camtoworlds                                # (N, 4, 4)
    Ks = [parser_obj.Ks_dict[cid].copy() for cid in parser_obj.camera_ids]
    widths = [parser_obj.imsize_dict[cid][0] for cid in parser_obj.camera_ids]
    heights = [parser_obj.imsize_dict[cid][1] for cid in parser_obj.camera_ids]

    splats = load_splats(args.ckpt, device)
    means = splats["means"]
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)             # (N, K, 3)
    sh_degree = int(math.sqrt(colors.shape[1]) - 1)

    from gsplat.rendering import rasterization

    rgbmaps, depthmaps = [], []
    for j in tqdm.tqdm(range(camtoworlds.shape[0]), desc="Rendering RGB and depth maps"):
        c2w = torch.from_numpy(camtoworlds[j].copy()[None]).float().to(device)
        K = torch.from_numpy(Ks[j][None]).float().to(device)
        with torch.no_grad():
            renders, _, _ = rasterization(
                means=means,
                quats=splats["quats"],
                scales=splats["scales"],
                opacities=splats["opacities"],
                colors=colors,
                viewmats=torch.linalg.inv(c2w),
                Ks=K,
                width=widths[j],
                height=heights[j],
                sh_degree=sh_degree,
                render_mode="RGB+ED",
                rasterize_mode="antialiased",
            )  # [1, H, W, 4]
        rgbmaps.append(torch.clamp(renders[0, :, :, 0:3].permute(2, 0, 1), 0.0, 1.0).cpu())
        depthmaps.append(renders[0, :, :, 3:4].permute(2, 0, 1).cpu())

    import open3d as o3d
    from gs.extract_mesh import (
        estimate_bounding_sphere,
        extract_mesh_bounded,
        extract_mesh_bounded_gpu,
        post_process_mesh,
    )

    center, radius = estimate_bounding_sphere(
        means.cpu().numpy(), camtoworlds, method="camera", gs_percentile=99, gs_scale=1.1,
    )
    print(f"[camera] Estimated bounding sphere: radius={radius:.2f}, center={center}")
    print(f"[camera] Use at least {radius * 2:.2f} for depth_trunc")

    print("Extracting bounded mesh ...")
    depth_trunc = (radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
    voxel_size = (depth_trunc / 128) if args.voxel_size < 0 else args.voxel_size
    # sdf_trunc = (4.0 * voxel_size) if args.sdf_trunc < 0 else args.sdf_trunc
    sdf_trunc = 0.3 if args.sdf_trunc < 0 else args.sdf_trunc
    print(f"  voxel_size={voxel_size}, sdf_trunc={sdf_trunc}, depth_trunc={depth_trunc}")

    if args.gpu_tsdf:
        mesh = extract_mesh_bounded_gpu(
            rgbmaps, depthmaps, camtoworlds, Ks, widths, heights,
            voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
            device_str=f"CUDA:{args.gpu_id}",
        )
    else:
        mesh = extract_mesh_bounded(
            rgbmaps, depthmaps, camtoworlds, Ks, widths, heights,
            voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
        )

    print("Post-processing mesh ...")
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    post_path = f"{args.out_dir}/fuse_post.ply"
    o3d.io.write_triangle_mesh(post_path, mesh_post)
    print(f"Saved {post_path}")

    if args.downsample_perc < 1.0:
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(post_path)
            ms.apply_filter(
                'meshing_decimation_quadric_edge_collapse',
                targetperc=args.downsample_perc,
                preservenormal=True,
            )
            ms.save_current_mesh(f"{args.out_dir}/fuse_simplified.ply")
            print(f"Saved {args.out_dir}/fuse_simplified.ply")
        except (ImportError, OSError) as e:
            print(f"[WARNING] pymeshlab unavailable ({e}), "
                  f"falling back to Open3D mesh simplification.")
            mesh_to_simplify = o3d.io.read_triangle_mesh(post_path)
            target_faces = int(len(mesh_to_simplify.triangles) * args.downsample_perc)
            mesh_simplified = mesh_to_simplify.simplify_quadric_decimation(
                target_number_of_triangles=target_faces)
            o3d.io.write_triangle_mesh(
                f"{args.out_dir}/fuse_simplified.ply", mesh_simplified)
            print(f"Saved {args.out_dir}/fuse_simplified.ply")


if __name__ == "__main__":
    main()
