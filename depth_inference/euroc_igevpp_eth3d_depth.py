#!/usr/bin/env python3
"""Generate cam0-aligned EuRoC depth maps with IGEV++ ETH3D weights."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder


DEFAULT_SEQUENCE = Path("/datasets/EuRoC/V203")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "pretrained_models/igev_plusplus/eth3d.pth"
DEFAULT_OUTPUT = DEFAULT_SEQUENCE / "depth_igevpp_eth3d"

IMAGE_SIZE = (752, 480)
K_LEFT = np.array(
    [[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
D_LEFT = np.array(
    [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05],
    dtype=np.float64,
)
K_RIGHT = np.array(
    [[457.587, 0.0, 379.999], [0.0, 456.134, 255.238], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
D_RIGHT = np.array(
    [-0.28368365, 0.07451284, -0.00010473, -3.555907e-05],
    dtype=np.float64,
)

# Rover-SLAM EuRoC.yaml Stereo.T_c1_c2. Rover-SLAM inverts this
# transformation before passing R and t to cv::stereoRectify.
T_LEFT_RIGHT = np.array(
    [
        [0.999997256477797, -0.002317135723275, -0.000343393120620, 0.110074137800478],
        [0.002312067192432, 0.999898048507103, -0.014090668452683, -0.000156612054392],
        [0.000376008102320, 0.014089835846691, 0.999900662638081, 0.000889382785432],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate rectified-cam0 EuRoC depth as 752x480 float32 meter NPY "
            "using IGEV++ ETH3D pretrained weights."
        )
    )
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--restore_ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--min_disp_px", type=float, default=0.5)
    parser.add_argument("--max_depth_m", type=float, default=0.0,
                        help="Set depth above this value to invalid; 0 disables the limit.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_preview", action="store_true")
    return parser.parse_args()


def make_model_args():
    return argparse.Namespace(
        mixed_precision=True,
        precision_dtype="float16",
        max_disp=768,
        valid_iters=16,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        n_downsample=2,
        s_disp_range=48,
        m_disp_range=96,
        l_disp_range=192,
        s_disp_interval=1,
        m_disp_interval=2,
        l_disp_interval=4,
        corr_radius=4,
        corr_levels=4,
        corr_implementation="reg",
        shared_backbone=False,
        dropout=0.0,
    )


def load_model(checkpoint, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = torch.nn.DataParallel(IGEVStereo(make_model_args()), device_ids=[0])
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.module.to(device)
    model.eval()
    return model


def build_rectification():
    right_to_left = np.linalg.inv(T_LEFT_RIGHT)
    rotation = right_to_left[:3, :3]
    translation = right_to_left[:3, 3]
    baseline_m = float(np.linalg.norm(T_LEFT_RIGHT[:3, 3]))

    r_left, r_right, p_left, p_right, _, _, _ = cv2.stereoRectify(
        K_LEFT,
        D_LEFT,
        K_RIGHT,
        D_RIGHT,
        IMAGE_SIZE,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1,
        newImageSize=IMAGE_SIZE,
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        K_LEFT, D_LEFT, r_left, p_left[:3, :3], IMAGE_SIZE, cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        K_RIGHT, D_RIGHT, r_right, p_right[:3, :3], IMAGE_SIZE, cv2.CV_32FC1
    )
    return map_lx, map_ly, map_rx, map_ry, p_left, p_right, baseline_m


def find_pairs(sequence):
    left_dir = sequence / "mav0/cam0/data"
    right_dir = sequence / "mav0/cam1/data"
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(
            f"Expected EuRoC image folders:\n  {left_dir}\n  {right_dir}"
        )
    right_by_name = {path.name: path for path in right_dir.glob("*.png")}
    pairs = [
        (left, right_by_name[left.name], left.stem)
        for left in sorted(left_dir.glob("*.png"))
        if left.name in right_by_name
    ]
    if not pairs:
        raise RuntimeError("No cam0/cam1 pairs with identical timestamp filenames")
    return pairs


def load_rectified_pair(left_path, right_path, maps, device):
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise RuntimeError(f"Failed to read stereo pair: {left_path}, {right_path}")
    if (left.shape[1], left.shape[0]) != IMAGE_SIZE or right.shape != left.shape:
        raise RuntimeError(f"Unexpected image shape: left={left.shape}, right={right.shape}")

    map_lx, map_ly, map_rx, map_ry = maps
    left = cv2.remap(left, map_lx, map_ly, cv2.INTER_LINEAR)
    right = cv2.remap(right, map_rx, map_ry, cv2.INTER_LINEAR)
    left_rgb = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)
    left_tensor = torch.from_numpy(left_rgb).permute(2, 0, 1).float()[None].to(device)
    right_tensor = torch.from_numpy(right_rgb).permute(2, 0, 1).float()[None].to(device)
    return left_tensor, right_tensor, left


@torch.no_grad()
def infer_disparity(model, left, right, valid_iters):
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    prediction = model(left, right, iters=valid_iters, test_mode=True)
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[-1]
    disparity = padder.unpad(prediction).squeeze()
    if disparity.ndim != 2:
        raise RuntimeError(f"Unexpected disparity shape: {tuple(disparity.shape)}")
    return disparity.float().cpu().numpy()


def disparity_to_depth(disparity, bf, min_disp_px, max_depth_m):
    depth = np.zeros(disparity.shape, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity >= min_disp_px)
    depth[valid] = bf / disparity[valid]
    depth[~np.isfinite(depth) | (depth <= 0.0)] = 0.0
    if max_depth_m > 0:
        depth[depth > max_depth_m] = 0.0
    return depth


def save_preview(path, depth):
    preview = np.zeros(depth.shape, dtype=np.uint8)
    valid = depth > 0
    if np.any(valid):
        near, far = np.percentile(depth[valid], [2, 98])
        if far > near:
            normalized = np.clip((depth - near) / (far - near), 0.0, 1.0)
            preview[valid] = np.round((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    if not cv2.imwrite(str(path), color):
        raise RuntimeError(f"Failed to save preview: {path}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.stride < 1 or args.start < 0:
        raise ValueError("--stride must be >= 1 and --start must be >= 0")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available in this container")

    pairs = find_pairs(args.sequence)
    all_pair_count = len(pairs)
    pairs = pairs[args.start::args.stride]
    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]
    if not pairs:
        raise RuntimeError("Frame selection produced zero stereo pairs")

    args.output.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output / "preview"
    if args.save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    map_lx, map_ly, map_rx, map_ry, p_left, p_right, baseline_m = build_rectification()
    fx_rect = float(p_left[0, 0])
    bf = fx_rect * baseline_m
    maps = (map_lx, map_ly, map_rx, map_ry)

    print(f"Sequence       : {args.sequence}")
    print(f"Output         : {args.output}")
    print(f"Matched pairs  : {all_pair_count}; selected: {len(pairs)}")
    print(f"Checkpoint     : {args.restore_ckpt}")
    print(f"Rectified fx   : {fx_rect:.12f} px")
    print(f"Baseline       : {baseline_m:.12f} m")
    print(f"bf             : {bf:.12f} px*m")
    print(f"Output format  : (480, 752), float32, meter, invalid=0")

    model = load_model(args.restore_ckpt, args.device)
    processed = 0
    skipped = 0
    total_time = 0.0
    rows = []

    for left_path, right_path, timestamp in tqdm(pairs, desc="IGEV++ ETH3D"):
        output_path = args.output / f"{timestamp}.npy"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        started = time.perf_counter()
        left, right, _ = load_rectified_pair(
            left_path, right_path, maps, args.device
        )
        disparity = infer_disparity(model, left, right, args.valid_iters)
        depth = disparity_to_depth(
            disparity, bf, args.min_disp_px, args.max_depth_m
        )
        if depth.shape != (480, 752) or depth.dtype != np.float32:
            raise RuntimeError(
                f"Invalid output for {timestamp}: shape={depth.shape}, dtype={depth.dtype}"
            )
        np.save(str(output_path), depth, allow_pickle=False)
        if args.save_preview:
            save_preview(preview_dir / f"{timestamp}.png", depth)

        elapsed = time.perf_counter() - started
        total_time += elapsed
        processed += 1
        rows.append(
            {
                "timestamp": timestamp,
                "file": output_path.name,
                "valid_ratio": float(np.count_nonzero(depth)) / depth.size,
                "runtime_sec": elapsed,
            }
        )

    metadata = {
        "sequence": str(args.sequence),
        "checkpoint": str(args.restore_ckpt),
        "checkpoint_sha256": sha256(args.restore_ckpt),
        "alignment": "rectified_cam0",
        "shape": [480, 752],
        "dtype": "float32",
        "unit": "meter",
        "invalid_value": 0.0,
        "matched_pairs": all_pair_count,
        "selected_pairs": len(pairs),
        "processed": processed,
        "skipped_existing": skipped,
        "fx_rect_px": fx_rect,
        "baseline_m": baseline_m,
        "bf_px_m": bf,
        "min_disp_px": args.min_disp_px,
        "max_depth_m": args.max_depth_m,
        "valid_iters": args.valid_iters,
        "p_left": p_left.tolist(),
        "p_right": p_right.tolist(),
        "frames": rows,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    average = total_time / processed if processed else 0.0
    print(
        f"Done: processed={processed}, skipped={skipped}, "
        f"average={average:.3f}s/frame, output={args.output}"
    )


if __name__ == "__main__":
    main()
