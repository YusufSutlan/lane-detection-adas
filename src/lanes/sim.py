"""Synthetic dashcam frames rendered from known lane geometry.

Ground truth for a lane: centre line x(y) = d + h*y + 0.5*k*y^2 (metres), width w.
d = lateral position of the lane centre relative to the vehicle, h = heading error,
k = curvature (1/m). The vehicle's lateral offset from the lane centre is -d.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as g

LINE_WIDTH_M = 0.15
Y_MAX = g.BEV_H / g.PX_PER_M_Y


@dataclass(frozen=True)
class LaneTruth:
    d: float
    h: float
    k: float
    width: float
    dashed_right: bool

    @property
    def offset(self) -> float:
        return -self.d

    def line_x(self, y: np.ndarray, side: int) -> np.ndarray:
        """side = -1 for the left line, +1 for the right line."""
        return self.d + self.h * y + 0.5 * self.k * y**2 + side * self.width / 2


def random_truth(rng: np.random.Generator) -> LaneTruth:
    return LaneTruth(
        d=rng.uniform(-0.4, 0.4), h=rng.uniform(-0.02, 0.02), k=rng.uniform(-0.006, 0.006),
        width=rng.uniform(3.2, 3.9), dashed_right=bool(rng.random() < 0.5),
    )  # fmt: skip


def _render_bev(truth: LaneTruth, fade: float, rng: np.random.Generator) -> np.ndarray:
    road = rng.normal(95, 6, (g.BEV_H, g.BEV_W)).astype(np.float32)
    road = cv2.GaussianBlur(road, (0, 0), 1.5)
    bev = np.dstack([road] * 3).astype(np.uint8)
    ys = np.linspace(0, Y_MAX, 200)
    color = int(95 + (235 - 95) * fade)
    thickness = max(1, int(LINE_WIDTH_M * g.PX_PER_M_X))
    for side in (-1, 1):
        pts = g.meters_to_bev(truth.line_x(ys, side), ys)
        if side == 1 and truth.dashed_right:
            # 3 m dashes every 8 m, phase randomised so dashes do not always start at the car.
            phase = rng.uniform(0, 8)
            for start in np.arange(-phase, Y_MAX, 8.0):
                sel = (ys >= start) & (ys <= start + 3.0)
                if sel.sum() > 1:
                    cv2.polylines(bev, [pts[sel].astype(np.int32)], False, (color,) * 3, thickness, cv2.LINE_AA)
        else:
            cv2.polylines(bev, [pts.astype(np.int32)], False, (color,) * 3, thickness, cv2.LINE_AA)
    return bev


def render(truth: LaneTruth, difficulty: float, rng: np.random.Generator) -> np.ndarray:
    """Render a grayscale camera frame. difficulty in [0, 1] scales noise, fading and shadows."""
    fade = 1.0 - 0.5 * difficulty
    bev = _render_bev(truth, fade, rng)
    img = cv2.warpPerspective(
        bev, g.BEV_TO_IMG, (g.IMG_W, g.IMG_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)
    img = img.mean(axis=2)
    img[: g.HORIZON] = np.linspace(190, 165, g.HORIZON)[:, None]  # sky

    # Shadows (e.g. trees, bridges): darken random polygons on the road.
    for _ in range(int(round(difficulty * 3))):
        pts = np.array(
            [[rng.uniform(0, g.IMG_W), rng.uniform(g.HORIZON, g.IMG_H)] for _ in range(4)], dtype=np.int32
        )
        mask = np.zeros(img.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
        img[mask == 1] *= rng.uniform(0.45, 0.75)

    img *= 1 + 0.15 * (rng.random() - 0.5)  # exposure
    img += rng.normal(0, 4 + 30 * difficulty, img.shape)  # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)
