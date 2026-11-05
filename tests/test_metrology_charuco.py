"""ChArUco calibration on rendered views of a known camera: the code has to
recover that camera, and distortion correction has to pay off on the plane."""
import cv2
import numpy as np
import pytest

from metrology import charuco_calib as cc


@pytest.fixture(scope="module")
def board():
    return cc.make_board()


@pytest.fixture(scope="module")
def intrinsics(board):
    return cc.calibrate_intrinsics(board, cc.demo_views(board, n=12))


def test_known_camera_is_recovered(intrinsics):
    K = np.array(intrinsics["camera_matrix"])
    assert K[0, 0] == pytest.approx(cc.DEMO_K[0, 0], rel=0.01)
    assert K[1, 1] == pytest.approx(cc.DEMO_K[1, 1], rel=0.01)
    assert K[0, 2] == pytest.approx(cc.DEMO_K[0, 2], abs=10)   # principal point is the weakest-determined term
    assert intrinsics["dist_coeffs"][0] == pytest.approx(cc.DEMO_DIST[0], abs=0.04)
    assert intrinsics["rms_px"] < 0.5
    assert intrinsics["heldout_rms_px"] is not None and intrinsics["heldout_rms_px"] < 0.8


def test_distortion_correction_cuts_plane_error(board, intrinsics):
    res = cc.plane_homography(board, cc.demo_plane_view(board), intrinsics)
    corrected, raw = res["corrected"], res["without_distortion_correction"]
    assert corrected["heldout_rms_mm"] < 0.5 * raw["heldout_rms_mm"]
    assert corrected["heldout_rms_mm"] < 0.1
    body = res["save_request"]
    assert len(body["pixel_points"]) == len(body["world_points"]) >= 10


def test_plane_frame_is_y_up(board, intrinsics):
    world = np.array(cc.plane_homography(board, cc.demo_plane_view(board), intrinsics)["save_request"]["world_points"])
    _, h = cc.board_size_mm(board)
    assert world[:, 1].min() > 0 and world[:, 1].max() < h
    # ChArUco corner ids run along rows from the top of the print; y-up means the first row has the largest y
    assert world[0, 1] > world[-1, 1]


def test_undistort_pixels_inverts_the_lens_model():
    obj = np.array([[x, y, 300.0] for x in (-80, 0, 80) for y in (-60, 0, 60)], dtype=np.float64)
    zero = np.zeros(3)
    distorted, _ = cv2.projectPoints(obj, zero, zero, cc.DEMO_K, cc.DEMO_DIST)
    ideal, _ = cv2.projectPoints(obj, zero, zero, cc.DEMO_K, None)
    back = cc.undistort_pixels(distorted.reshape(-1, 2), cc.DEMO_K, cc.DEMO_DIST)
    assert np.abs(back - ideal.reshape(-1, 2)).max() < 1e-3


def test_four_points_give_zero_residual_and_no_heldout_number():
    res = cc.homography_residuals([[0, 0], [400, 0], [400, 300], [0, 300]], [[0, 0], [100, 0], [100, 75], [0, 75]])
    assert res["rms_mm"] < 1e-9 and res["heldout_rms_mm"] is None


def test_collinear_points_are_rejected():
    with pytest.raises(ValueError):
        cc.homography_residuals([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 1], [2, 2], [3, 3]])
