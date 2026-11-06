#!/usr/bin/env python3
"""Convert COLMAP sparse/0 model -> WorldMirror prior camera_params.json.

Usage:
    python utils/colmap_to_prior.py <scene_sparse_dir> <images_dir> <out_json>
        [--max_frames 32]

Only frames actually present in <images_dir> are exported (image-dir may be a
subsampled copy). camera_id = image filename stem (matches pipeline docs).
"""
import argparse, json, struct
from pathlib import Path

import numpy as np

# COLMAP camera param layouts we support
CAMERA_PARAMS = {
    0: ("SIMPLE_PINHOLE", ["f", "cx", "cy"]),
    1: ("PINHOLE", ["fx", "fy", "cx", "cy"]),
    2: ("SIMPLE_RADIAL", ["f", "cx", "cy", "k"]),
    3: ("RADIAL", ["f", "cx", "cy", "k1", "k2"]),
    4: ("OPENCV", ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]),
    5: ("OPENCV_FISHEYE", ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]),
    9: ("FOV", ["fx", "fy", "cx", "cy", "omega"]),
    10: ("SIMPLE_RADIAL_FISHEYE", ["f", "cx", "cy", "k"]),
    11: ("RADIAL_FISHEYE", ["f", "cx", "cy", "k1", "k2"]),
}


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iIQQ", f.read(24))
            nparams = len(CAMERA_PARAMS.get(model_id, (None, []))[1])
            if model_id not in CAMERA_PARAMS:
                raise ValueError(f"unsupported camera model_id={model_id}")
            params = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
            cams[cam_id] = dict(zip(CAMERA_PARAMS[model_id][1], params))
    return cams


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_images(path):
    """return {image_stem: (w2c 4x4, camera_id)}"""
    imgs = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)  # skip xys + point3D_ids
            R = qvec2rot(qvec)
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = R, np.array(tvec)
            imgs[Path(name.decode()).stem] = (w2c, cam_id)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sparse_dir", help="scene sparse/0 dir")
    ap.add_argument("images_dir", help="dir of images actually fed to pipeline")
    ap.add_argument("out_json")
    args = ap.parse_args()

    cams = read_cameras(Path(args.sparse_dir) / "cameras.bin")
    imgs = read_images(Path(args.sparse_dir) / "images.bin")

    stems = sorted(p.stem for p in Path(args.images_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    extrinsics, intrinsics = [], []
    for s in stems:
        if s not in imgs:
            print(f"warn: {s} not in images.bin, skipped")
            continue
        w2c, cam_id = imgs[s]
        c2w = np.linalg.inv(w2c)
        p = cams[cam_id]
        fx = p.get("fx", p.get("f"))
        fy = p.get("fy", p.get("f"))
        K = [[fx, 0.0, p["cx"]], [0.0, fy, p["cy"]], [0.0, 0.0, 1.0]]
        extrinsics.append({"camera_id": s, "matrix": c2w.tolist()})
        intrinsics.append({"camera_id": s, "matrix": K})

    out = {"num_cameras": len(extrinsics),
           "extrinsics": extrinsics, "intrinsics": intrinsics}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    print(f"{args.out_json}: {len(extrinsics)} cams")


if __name__ == "__main__":
    main()
