import numpy as np

from lanes import geometry as g
from lanes.detect import LaneEstimate, detect_lanes
from lanes.experiment import evaluate
from lanes.sim import LaneTruth, random_truth, render


def test_homography_roundtrip():
    pts = np.array([[100.0, 50.0], [300.0, 350.0]])
    back = g.bev_to_image(pts)
    assert back.shape == (2, 2)
    import cv2

    again = cv2.perspectiveTransform(back.reshape(-1, 1, 2).astype(np.float32), g.IMG_TO_BEV).reshape(-1, 2)
    assert np.allclose(again, pts, atol=1e-2)


def test_meter_pixel_conversion():
    px = g.meters_to_bev(np.array([0.0, 1.0]), np.array([0.0, 10.0]))
    x, y = g.bev_to_meters(px[:, 0], px[:, 1])
    assert np.allclose(x, [0, 1]) and np.allclose(y, [0, 10])


def test_render_is_deterministic_per_seed():
    t = random_truth(np.random.default_rng(1))
    a = render(t, 0.5, np.random.default_rng(2))
    b = render(t, 0.5, np.random.default_rng(2))
    assert np.array_equal(a, b) and a.shape == (g.IMG_H, g.IMG_W)


def test_clean_scene_is_measured_accurately():
    truth = LaneTruth(d=0.3, h=0.01, k=0.004, width=3.6, dashed_right=False)
    est = detect_lanes(render(truth, 0.0, np.random.default_rng(0)))
    assert est is not None
    assert abs(est.offset - truth.offset) < 0.05
    assert abs(est.curvature - truth.k) < 0.001
    assert abs(est.width - truth.width) < 0.1


def test_blank_image_returns_none():
    assert detect_lanes(np.full((g.IMG_H, g.IMG_W), 90, np.uint8)) is None


def test_estimate_properties():
    est = LaneEstimate(np.array([0.002, 0.0, -1.8]), np.array([0.002, 0.0, 1.8]), 500, 500)
    assert est.offset == 0 and abs(est.curvature - 0.004) < 1e-12 and est.width == 3.6


def test_benchmark_degrades_gracefully():
    easy = evaluate(0.0, n=20)
    hard = evaluate(1.0, n=20)
    assert easy["accurate"] >= 0.95 and easy["offset_mae_cm"] < 3
    assert hard["offset_mae_cm"] >= easy["offset_mae_cm"]
