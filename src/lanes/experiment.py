"""Benchmark lane detection over increasing difficulty and draw qualitative figures.

Usage:
    python -m lanes.experiment
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import geometry as g
from .detect import detect_lanes
from .sim import Y_MAX, random_truth, render

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
OFFSET_TOL_M = 0.15


def evaluate(difficulty: float, n: int, seed: int = 0, img_to_bev: np.ndarray = g.IMG_TO_BEV) -> dict:
    rng = np.random.default_rng(seed)
    off_err, curv_err, width_err, ok, failed = [], [], [], 0, 0
    for _ in range(n):
        truth = random_truth(rng)
        est = detect_lanes(render(truth, difficulty, rng), img_to_bev=img_to_bev)
        if est is None:
            failed += 1
            continue
        off_err.append(abs(est.offset - truth.offset))
        curv_err.append(abs(est.curvature - truth.k))
        width_err.append(abs(est.width - truth.width))
        ok += off_err[-1] < OFFSET_TOL_M
    return {
        "difficulty": difficulty,
        "detected": 1 - failed / n,
        "accurate": ok / n,  # detected AND lateral offset within 15 cm
        "offset_mae_cm": 100 * float(np.mean(off_err)) if off_err else float("nan"),
        "offset_p95_cm": 100 * float(np.percentile(off_err, 95)) if off_err else float("nan"),
        "curvature_mae_per_km": 1000 * float(np.mean(curv_err)) if curv_err else float("nan"),
        "width_mae_cm": 100 * float(np.mean(width_err)) if width_err else float("nan"),
    }


def draw_overlay(gray: np.ndarray, est) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if est is None:
        return out
    ys = np.linspace(0, Y_MAX, 40)
    left = g.bev_to_image(g.meters_to_bev(np.polyval(est.left, ys), ys))
    right = g.bev_to_image(g.meters_to_bev(np.polyval(est.right, ys), ys))
    poly = np.vstack([left, right[::-1]]).astype(np.int32)
    layer = out.copy()
    cv2.fillPoly(layer, [poly], (0, 200, 0))
    out = cv2.addWeighted(layer, 0.35, out, 0.65, 0)
    cv2.polylines(out, [left.astype(np.int32)], False, (0, 0, 255), 2)
    cv2.polylines(out, [right.astype(np.int32)], False, (255, 0, 0), 2)
    return out


def qualitative_figure(seed: int = 4) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5))
    for row, difficulty in enumerate([0.0, 0.5, 1.0]):
        rng = np.random.default_rng(seed + row)
        truth = random_truth(rng)
        gray = render(truth, difficulty, rng)
        est, dbg = detect_lanes(gray, return_debug=True)
        axes[row, 0].imshow(draw_overlay(gray, est)[..., ::-1])
        title = "no lane found" if est is None else f"offset err {100*abs(est.offset-truth.offset):.0f} cm"
        axes[row, 0].set_title(f"difficulty {difficulty:.1f}: {title}")
        axes[row, 1].imshow(dbg["bev"], cmap="gray")
        axes[row, 1].set_title("bird's-eye view")
        vis = np.dstack([dbg["mask"] * 255] * 3)
        for (xs, ys), color in ((dbg["left_pixels"], (255, 60, 60)), (dbg["right_pixels"], (60, 120, 255))):
            vis[ys, xs] = color
        axes[row, 2].imshow(vis)
        axes[row, 2].set_title("line mask + pixels used for the fit")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "qualitative.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'difficulty':>10}{'detected':>10}{'accurate':>10}{'offset MAE cm':>15}{'p95 cm':>9}{'curv MAE /km':>14}")
    for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = evaluate(d, n=100)
        rows.append(r)
        print(f"{d:>10.2f}{r['detected']:>10.2f}{r['accurate']:>10.2f}{r['offset_mae_cm']:>15.1f}"
              f"{r['offset_p95_cm']:>9.1f}{r['curvature_mae_per_km']:>14.2f}")  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = [r["difficulty"] for r in rows]
    axes[0].plot(x, [r["detected"] for r in rows], "o-", label="lane found")
    axes[0].plot(x, [r["accurate"] for r in rows], "s-", label=f"found and offset < {int(OFFSET_TOL_M*100)} cm")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("difficulty (noise + faded paint + shadows)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, [r["offset_mae_cm"] for r in rows], "o-", label="offset MAE (cm)")
    axes[1].plot(x, [r["offset_p95_cm"] for r in rows], "s--", label="offset p95 (cm)")
    axes[1].set_xlabel("difficulty")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Lane detection vs. difficulty (100 random scenes per level)")
    fig.tight_layout()
    fig.savefig(FIGURES / "difficulty_sweep.png", dpi=130)
    plt.close(fig)

    qualitative_figure()

    print("\nCalibration sensitivity (difficulty 0.25): camera pitch error shifts the horizon")
    calib = []
    for shift in [-6, -3, 0, 3, 6]:
        r = evaluate(0.25, n=100, img_to_bev=g.miscalibrated_img_to_bev(shift))
        calib.append({"horizon_shift_px": shift, **r})
        print(f"  horizon {shift:+d} px: detected {r['detected']:.2f}, offset MAE {r['offset_mae_cm']:.1f} cm, "
              f"curvature MAE {r['curvature_mae_per_km']:.2f} /km")  # fmt: skip
    (ROOT / "reports" / "calibration_sensitivity.json").write_text(json.dumps(calib, indent=2))

    (ROOT / "reports" / "benchmark.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
