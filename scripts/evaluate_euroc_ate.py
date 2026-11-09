#!/usr/bin/env python3
"""Evaluate a TUM-format estimated trajectory against EuRoC ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_euroc_gt(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            rows.append([
                float(parts[0]) * 1e-9,
                float(parts[1]), float(parts[2]), float(parts[3]),
                float(parts[5]), float(parts[6]), float(parts[7]), float(parts[4]),
            ])
    return np.asarray(rows, dtype=float)


def load_tum(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append([float(x) for x in parts[:8]])
    return np.asarray(rows, dtype=float)


def associate(gt: np.ndarray, est: np.ndarray, max_dt: float):
    g_times = gt[:, 0]
    g_idx = []
    e_idx = []
    i = 0
    for j, t in enumerate(est[:, 0]):
        while i + 1 < len(g_times) and abs(g_times[i + 1] - t) <= abs(g_times[i] - t):
            i += 1
        if abs(g_times[i] - t) <= max_dt:
            g_idx.append(i)
            e_idx.append(j)
    return gt[g_idx], est[e_idx]


def umeyama(model: np.ndarray, data: np.ndarray, align_scale: bool):
    mu_m = model.mean(axis=1, keepdims=True)
    mu_d = data.mean(axis=1, keepdims=True)
    xm = model - mu_m
    xd = data - mu_d
    w = xd @ xm.T / model.shape[1]
    u, s, vt = np.linalg.svd(w)
    sfix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sfix[2, 2] = -1
    rot = u @ sfix @ vt
    if align_scale:
        var = np.sum(xm * xm) / model.shape[1]
        scale = float(np.trace(np.diag(s) @ sfix) / var)
    else:
        scale = 1.0
    trans = mu_d - scale * rot @ mu_m
    aligned = scale * rot @ model + trans
    return aligned, scale, rot, trans.reshape(3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SE(3) or Sim(3) ATE RMSE against EuRoC MAV ground truth."
    )
    parser.add_argument("--gt", type=Path, required=True, help="Path to EuRoC state_groundtruth_estimate0/data.csv")
    parser.add_argument("--traj", type=Path, required=True, help="Path to estimated trajectory in TUM format (f_*.txt)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional directory to save metrics.json, metrics.txt, and per_pose_ate.csv")
    parser.add_argument("--max-dt", type=float, default=0.05, help="Max timestamp association difference in seconds (default: 0.05)")
    parser.add_argument("--align-scale", action="store_true", help="Use Sim(3) scale alignment instead of strict SE(3)")
    args = parser.parse_args()

    gt = load_euroc_gt(args.gt)
    est = load_tum(args.traj)
    if len(gt) == 0:
        raise SystemExit(f"Ground truth is empty: {args.gt}")
    if len(est) == 0:
        raise SystemExit(f"Trajectory is empty: {args.traj}")

    g, e = associate(gt, est, args.max_dt)
    if len(g) < 10:
        raise SystemExit(f"Too few associated poses: {len(g)}")

    gp = g[:, 1:4].T
    ep = e[:, 1:4].T
    aligned, scale, _, _ = umeyama(ep, gp, args.align_scale)
    ate = np.linalg.norm(gp - aligned, axis=0)

    metrics = {
        "gt_csv": str(args.gt),
        "trajectory_file": str(args.traj),
        "association_max_time_diff_sec": args.max_dt,
        "alignment": "Sim3_scale" if args.align_scale else "SE3_no_scale",
        "scale": scale,
        "gt_poses": int(len(gt)),
        "estimated_poses": int(len(est)),
        "associated_poses": int(len(ate)),
        "ate_rmse_m": float(math.sqrt(np.mean(ate * ate))),
        "ate_mean_m": float(np.mean(ate)),
        "ate_median_m": float(np.median(ate)),
        "ate_std_m": float(np.std(ate)),
        "ate_min_m": float(np.min(ate)),
        "ate_max_m": float(np.max(ate)),
    }
    print(json.dumps(metrics, indent=2))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (args.out_dir / "per_pose_ate.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_est", "timestamp_gt", "dt_sec",
                "gt_x", "gt_y", "gt_z",
                "est_aligned_x", "est_aligned_y", "est_aligned_z",
                "ate_m",
            ])
            for gi, ei, ai, err in zip(g, e, aligned.T, ate):
                writer.writerow([
                    f"{ei[0]:.9f}", f"{gi[0]:.9f}", f"{ei[0] - gi[0]:.9f}",
                    f"{gi[1]:.9f}", f"{gi[2]:.9f}", f"{gi[3]:.9f}",
                    f"{ai[0]:.9f}", f"{ai[1]:.9f}", f"{ai[2]:.9f}",
                    f"{err:.9f}",
                ])


if __name__ == "__main__":
    main()

