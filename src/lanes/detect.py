"""Lane detection: inverse perspective mapping, top-hat line extraction, sliding windows,
quadratic fit in metres, and sanity checks that reject implausible results."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as g

N_WINDOWS, MARGIN, MIN_PIX = 10, 38, 25


@dataclass
class LaneEstimate:
    left: np.ndarray  # quadratic coefficients (A, B, C): x = A*y^2 + B*y + C, metres
    right: np.ndarray
    n_left: int
    n_right: int

    @property
    def offset(self) -> float:
        return -(self.left[2] + self.right[2]) / 2

    @property
    def curvature(self) -> float:
        return float(self.left[0] + self.right[0])  # 2A of the centre line = k

    @property
    def width(self) -> float:
        return float(self.right[2] - self.left[2])


def to_bev(gray: np.ndarray, img_to_bev: np.ndarray = g.IMG_TO_BEV) -> np.ndarray:
    return cv2.warpPerspective(gray, img_to_bev, (g.BEV_W, g.BEV_H), flags=cv2.INTER_LINEAR)


def extract_lines(bev: np.ndarray) -> np.ndarray:
    """Binary mask of thin bright structures. A top-hat filter is a local operator, so a
    shadow edge (a large step) is suppressed while a ~7 px wide painted line survives."""
    blur = cv2.GaussianBlur(bev, (0, 0), 1.5)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    thr, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (tophat > max(thr, 18)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _sliding_window(mask: np.ndarray, x_start: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = mask.nonzero()
    win_h = g.BEV_H // N_WINDOWS
    x_cur, kept = x_start, []
    for w in range(N_WINDOWS):
        y_hi, y_lo = g.BEV_H - (w + 1) * win_h, g.BEV_H - w * win_h
        sel = np.where((ys >= y_hi) & (ys < y_lo) & (np.abs(xs - x_cur) < MARGIN))[0]
        kept.append(sel)
        if len(sel) >= MIN_PIX:
            x_cur = int(xs[sel].mean())
    idx = np.concatenate(kept)
    return xs[idx], ys[idx]


def _fit(xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
    if len(xs) < 150:
        return None
    x_m, y_m = g.bev_to_meters(xs.astype(float), ys.astype(float))
    coef = np.polyfit(y_m, x_m, 2)
    keep = np.abs(np.polyval(coef, y_m) - x_m) < 0.25  # drop outliers and refit
    if keep.sum() < 100:
        return None
    return np.polyfit(y_m[keep], x_m[keep], 2)


def detect_lanes(gray: np.ndarray, return_debug: bool = False, img_to_bev: np.ndarray = g.IMG_TO_BEV):
    bev = to_bev(gray, img_to_bev)
    mask = extract_lines(bev)

    # Start each line from a column-histogram peak in the nearest part of the BEV.
    hist = mask[g.BEV_H * 2 // 3 :].sum(axis=0).astype(float)
    hist = np.convolve(hist, np.ones(9), mode="same")
    mid = g.BEV_W // 2
    left_x, right_x = int(np.argmax(hist[:mid])), mid + int(np.argmax(hist[mid:]))
    debug = {"bev": bev, "mask": mask}

    lx, ly = _sliding_window(mask, left_x)
    rx, ry = _sliding_window(mask, right_x)
    left, right = _fit(lx, ly), _fit(rx, ry)
    debug.update(left_pixels=(lx, ly), right_pixels=(rx, ry))
    if left is None or right is None:
        return (None, debug) if return_debug else None

    est = LaneEstimate(left, right, len(lx), len(rx))
    far = np.array([5.0, 12.0])
    widths = np.polyval(right, far) - np.polyval(left, far)
    plausible = 2.8 < est.width < 4.5 and np.all((widths > 2.8) & (widths < 4.5))
    result = est if plausible else None
    return (result, debug) if return_debug else result
