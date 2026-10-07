"""Shared geometry: bird's-eye-view (BEV) grid and the BEV <-> camera homography.

BEV is a top-down grid centred on the vehicle: 50 px per metre laterally, 20 px per metre
ahead, row 0 = 20 m ahead, last row = vehicle position. The camera homography stands in for
an extrinsic calibration (known from the vehicle's mounting), as in a production ADAS.
"""

from __future__ import annotations

import cv2
import numpy as np

BEV_W, BEV_H = 440, 400
PX_PER_M_X, PX_PER_M_Y = 50.0, 20.0
IMG_W, IMG_H = 640, 360
HORIZON = 185

_BEV_CORNERS = np.float32([[0, 0], [BEV_W, 0], [BEV_W, BEV_H], [0, BEV_H]])
_IMG_CORNERS = np.float32([[255, HORIZON], [385, HORIZON], [840, IMG_H - 1], [-200, IMG_H - 1]])

BEV_TO_IMG = cv2.getPerspectiveTransform(_BEV_CORNERS, _IMG_CORNERS)
IMG_TO_BEV = np.linalg.inv(BEV_TO_IMG)


def meters_to_bev(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """(lateral x, distance ahead y) in metres -> BEV pixel coordinates, shape (N, 2)."""
    return np.stack([BEV_W / 2 + x_m * PX_PER_M_X, BEV_H - y_m * PX_PER_M_Y], axis=-1)


def bev_to_meters(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (px - BEV_W / 2) / PX_PER_M_X, (BEV_H - py) / PX_PER_M_Y


def bev_to_image(points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), BEV_TO_IMG).reshape(-1, 2)


def miscalibrated_img_to_bev(horizon_shift_px: float) -> np.ndarray:
    """IMG -> BEV homography as if the camera pitch were wrong: the horizon row is shifted."""
    corners = _IMG_CORNERS.copy()
    corners[:2, 1] += horizon_shift_px
    return np.linalg.inv(cv2.getPerspectiveTransform(_BEV_CORNERS, corners))
