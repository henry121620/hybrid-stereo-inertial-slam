#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASETS = {
    "V101": "/datasets/EuRoC/V1_01_easy",
    "V102": "/datasets/EuRoC/V1_02_medium",
    "V103": "/datasets/EuRoC/V1_03_difficult",
    "V201": "/datasets/EuRoC/V2_01_easy",
    "V202": "/datasets/EuRoC/V2_02_medium",
    "V203": "/datasets/EuRoC/V203",
    "MH01": "/datasets/EuRoC/MH01",
    "MH02": "/datasets/EuRoC/MH02",
    "MH03": "/datasets/EuRoC/MH03",
    "MH04": "/datasets/EuRoC/MH04",
    "MH05": "/datasets/EuRoC/MH05",
}


EXPECTED_MATCHED = {
    "V101": 2790,
    "V102": 1588,
    "V103": 1995,
    "V201": 2180,
    "V202": 2263,
    "V203": 1805,
    "MH01": 3682,
    "MH02": 3040,
    "MH03": 2633,
    "MH04": 2033,
    "MH05": 2273,
}


def trajectory_quality(seq: str, matched: object, kfs: str) -> tuple[bool, str]:
    try:
        matched_i = int(matched)
    except Exception:
        return False, "missing_matched"
    min_matched = int(EXPECTED_MATCHED.get(seq, 100) * 0.5)
    if matched_i < min_matched:
        return False, f"short_trajectory_matched_{matched_i}_lt_{min_matched}"
    m = re.search(r"Map 0 has\s+(\d+)\s+KFs", kfs or "")
    if m and int(m.group(1)) < 20:
        return False, f"too_few_kfs_{m.group(1)}"
    return True, ""


def load_tum(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append([float(x) for x in parts[:8]])
    return np.asarray(rows, dtype=float)


def load_euroc_gt(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 8:
                ts = float(parts[0]) * 1e-9
                rows.append([ts, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5]), float(parts[6]), float(parts[7]), float(parts[4])])
    return np.asarray(rows, dtype=float)


def associate(gt: np.ndarray, est: np.ndarray, max_dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
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


def umeyama_align(model: np.ndarray, data: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
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
    var = np.sum(xm * xm) / model.shape[1]
    scale = float(np.trace(np.diag(s) @ sfix) / var)
    trans = mu_d - scale * rot @ mu_m
    return scale, rot, trans


def rmse_for(seq: str, traj: Path) -> tuple[float, int]:
    gt_tum = Path(DATASETS[seq]) / "mav0/state_groundtruth_estimate0/data.tum"
    gt_csv = Path(DATASETS[seq]) / "mav0/state_groundtruth_estimate0/data.csv"
    gt = load_tum(gt_tum) if gt_tum.exists() else load_euroc_gt(gt_csv)
    est = load_tum(traj)
    g, e = associate(gt, est)
    if len(g) < 10:
        raise RuntimeError(f"too few matched poses: {len(g)}")
    gp = g[:, 1:4].T
    ep = e[:, 1:4].T
    scale, rot, trans = umeyama_align(ep, gp)
    aligned = scale * rot @ ep + trans
    err = np.linalg.norm(gp - aligned, axis=0)
    return math.sqrt(float(np.mean(err * err))), len(g)


def read_manifest(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    path = run_dir / "run_manifest.txt"
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_evidence_csv(path: Path) -> dict[str, float | int]:
    if not path.exists() or path.stat().st_size == 0:
        return {"csv_rows": 0}
    sums: dict[str, float] = defaultdict(float)
    count = 0
    fields = (
        "injected",
        "replaced_high",
        "replaced_mid",
        "conflict_keep_stereo",
        "source_stereo_only",
        "source_igev_only",
        "source_fused_high",
        "source_conflict_stereo",
        "source_fused_mid",
        "source_suppressed",
        "calib_used_pairs",
        "calib_median_rel_before",
        "calib_median_rel_after",
        "calib_p90_rel_before",
        "calib_p90_rel_after",
        "avg_conf",
        "occupied_cells",
        "empty_cells",
        "cell_cv",
        "grid_entropy_norm",
        "conservative_quota_avg",
        "adaptive_quota_suppressed",
        "recovery_cells",
    )
    with path.open("r", newline="", errors="ignore") as f:
        for row in csv.DictReader(f):
            count += 1
            for field in fields:
                value = row.get(field, "")
                if value:
                    try:
                        sums[field] += float(value)
                    except ValueError:
                        pass
    out: dict[str, float | int] = {"csv_rows": count}
    for field in fields:
        out[f"{field}_avg"] = sums[field] / count if count else float("nan")
    return out


def parse_log(run_dir: Path, run_name: str) -> dict[str, int | str]:
    logs = list(run_dir.glob("*.log"))
    log_path = run_dir / f"{run_name}.log" if run_name else (logs[0] if logs else Path(""))
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    kfs = ""
    for line in text.splitlines():
        if "Map 0 has" in line:
            kfs = line.strip()
    return {
        "lost": text.count("Lost for a short time"),
        "loops": text.count("Loop detected"),
        "not_enough": text.count("not enough acceleration"),
        "fail_local": text.count("Fail to track local map"),
        "resets": text.count("Active map Reseting"),
        "igev_logs": text.count("[IGEV D4 REAL]"),
        "edge_logs": text.count("edge_aware_enabled=1"),
        "kfs": kfs,
    }


def status_of(run_dir: Path) -> str:
    path = run_dir / "run_status.txt"
    return path.read_text(errors="ignore").strip() if path.exists() else "missing"


def safe_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def aggregate(rows: list[dict[str, object]], key_fields: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in key_fields)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        rmses = np.asarray([safe_float(r.get("rmse")) for r in group if not math.isnan(safe_float(r.get("rmse")))], dtype=float)
        lost = np.asarray([safe_float(r.get("lost")) for r in group], dtype=float)
        item = {field: value for field, value in zip(key_fields, key)}
        item["runs"] = len(group)
        if len(rmses):
            item["rmse_mean"] = float(np.mean(rmses))
            item["rmse_median"] = float(np.median(rmses))
            item["rmse_std"] = float(np.std(rmses))
            item["rmse_best"] = float(np.min(rmses))
            item["rmse_worst"] = float(np.max(rmses))
        item["lost_mean"] = float(np.mean(lost)) if len(lost) else float("nan")
        for field in ("injected_avg", "source_igev_only_avg", "grid_entropy_norm_avg", "cell_cv_avg", "calib_median_rel_before_avg", "calib_median_rel_after_avg"):
            vals = np.asarray([safe_float(r.get(field)) for r in group if not math.isnan(safe_float(r.get(field)))], dtype=float)
            item[field] = float(np.mean(vals)) if len(vals) else float("nan")
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/runs/igev_work_evidence")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-prefix", default=None)
    ap.add_argument("--valid-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    rows: list[dict[str, object]] = []
    for run_dir in sorted(root.glob("*/*")):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        run_id = run_dir.name[4:] if run_dir.name.startswith("run_") else run_dir.name
        if args.run_id and run_id != args.run_id:
            continue
        if args.run_prefix and not run_id.startswith(args.run_prefix):
            continue
        manifest = read_manifest(run_dir)
        seq = manifest.get("seq", "")
        run_name = manifest.get("run_name", "")
        row: dict[str, object] = {
            "seq": seq,
            "profile": manifest.get("profile", ""),
            "variant": manifest.get("variant", "FULL"),
            "logging": manifest.get("evidence_logging", ""),
            "run": run_id,
            "status": status_of(run_dir),
            "run_dir": str(run_dir),
        }
        trajs = list(run_dir.glob("f_*.txt"))
        if seq in DATASETS and trajs:
            try:
                row["rmse"], row["matched"] = rmse_for(seq, trajs[0])
            except Exception as exc:
                row["rmse_error"] = str(exc)
        row.update(parse_log(run_dir, run_name))
        valid, invalid_reason = trajectory_quality(seq, row.get("matched"), str(row.get("kfs", "")))
        row["trajectory_valid"] = int(valid)
        row["invalid_reason"] = invalid_reason
        frame_csv = manifest.get("frame_csv", "")
        if frame_csv:
            row.update(parse_evidence_csv(Path(frame_csv)))
        rows.append(row)

    if args.valid_only:
        rows = [row for row in rows if row.get("trajectory_valid") == 1]

    summary_dir = root / "_summary"
    write_csv(summary_dir / "evidence_run_summary.csv", rows)
    write_csv(summary_dir / "evidence_variant_summary.csv", aggregate(rows, ("seq", "profile", "variant")))
    write_csv(summary_dir / "logging_intrusion_summary.csv", aggregate(rows, ("seq", "profile", "variant", "logging")))

    display_fields = ("seq", "profile", "variant", "logging", "run", "status", "trajectory_valid", "invalid_reason", "rmse", "matched", "lost", "loops", "csv_rows", "injected_avg", "source_igev_only_avg", "grid_entropy_norm_avg", "calib_median_rel_before_avg", "calib_median_rel_after_avg", "kfs")
    print(",".join(display_fields))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in display_fields))
    print()
    print(f"wrote {summary_dir / 'evidence_run_summary.csv'}")
    print(f"wrote {summary_dir / 'evidence_variant_summary.csv'}")
    print(f"wrote {summary_dir / 'logging_intrusion_summary.csv'}")


if __name__ == "__main__":
    main()
