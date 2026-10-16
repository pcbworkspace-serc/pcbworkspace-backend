"""/calibration/save with camera intrinsics: points are undistorted before
the homography, residuals are reported, and the old 4-point request still
works but says what it can't check."""
import cv2
import numpy as np

from metrology.charuco_calib import DEMO_DIST, DEMO_K


def _plane_points():
    world = np.array([(x, y) for x in np.linspace(0, 80, 9) for y in np.linspace(0, 60, 7)], dtype=np.float64)
    obj = np.column_stack([world, np.zeros(len(world))])
    rvec, tvec = np.array([0.1, -0.08, 0.02]), np.array([-40.0, -30.0, 130.0])
    pix, _ = cv2.projectPoints(obj, rvec, tvec, DEMO_K, DEMO_DIST)
    return pix.reshape(-1, 2).tolist(), world.tolist()


def test_intrinsics_remove_lens_distortion_from_the_homography(client):
    pix, world = _plane_points()
    try:
        raw = client.post("/calibration/save", json={"pixel_points": pix, "world_points": world}).get_json()
        corrected = client.post("/calibration/save", json={
            "pixel_points": pix, "world_points": world,
            "intrinsics": {"camera_matrix": DEMO_K.tolist(), "dist_coeffs": DEMO_DIST.tolist()},
        }).get_json()
        assert corrected["distortion_corrected"] and not raw["distortion_corrected"]
        assert corrected["residual"]["rms_mm"] < 0.01
        assert raw["residual"]["heldout_rms_mm"] > 10 * corrected["residual"]["heldout_rms_mm"]
        got = client.get("/calibration/get").get_json()
        assert got["distortion_corrected"] and got["intrinsics"]["camera_matrix"] == DEMO_K.tolist()
    finally:
        client.post("/calibration/clear")
    assert client.get("/calibration/get").status_code == 404


def test_four_point_calibration_says_its_residual_checks_nothing(client):
    try:
        r = client.post("/calibration/save", json={
            "pixel_points": [[0, 0], [400, 0], [400, 300], [0, 300]],
            "world_points": [[0, 0], [100, 0], [100, 75], [0, 75]],
        }).get_json()
        assert r["ok"] and r["residual"]["rms_mm"] < 1e-9
        assert "by construction" in r["residual"]["note"]
    finally:
        client.post("/calibration/clear")


def test_malformed_intrinsics_and_points_are_rejected(client):
    pix, world = _plane_points()
    bad_k = client.post("/calibration/save", json={"pixel_points": pix, "world_points": world,
                                                   "intrinsics": {"camera_matrix": [[1, 0], [0, 1]], "dist_coeffs": [0, 0, 0, 0]}})
    assert bad_k.status_code == 400
    bad_pts = client.post("/calibration/save", json={"pixel_points": [[0, 0, 0]] * 4, "world_points": [[0, 0, 0]] * 4})
    assert bad_pts.status_code == 400
    collinear = client.post("/calibration/save", json={"pixel_points": [[0, 0], [1, 1], [2, 2], [3, 3]],
                                                       "world_points": [[0, 0], [1, 1], [2, 2], [3, 3]]})
    assert collinear.status_code == 400


def test_status_says_whether_arm_geometry_is_measured(client):
    reach = client.get("/nn/status").get_json()["robot_safety"]["kinematic_reach_check"]
    assert reach["arm_geometry"] == "placeholder"
    assert "unmeasured placeholder" in reach["note"]
