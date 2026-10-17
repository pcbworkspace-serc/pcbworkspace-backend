"""
charuco_calib.py — camera intrinsics + lens distortion from a ChArUco target,
then the board-plane homography on distortion-corrected points.

The backend's /calibration/save fits a homography to raw pixel points. A
homography can't model lens distortion, so on a low-cost USB lens the
error grows toward the image edges. This fixes it in the usual order:

1. Print the target at 100% scale and MEASURE one square with calipers;
   use the measured size, not the nominal one.
       python -m metrology.charuco_calib make-board --out charuco.png
2. Intrinsics: 15-30 photos of the target, tilted up to ~40 degrees, covering
   the corners and edges of the image as well as the middle.
       python -m metrology.charuco_calib intrinsics photos/ --square-mm 10.02 --out top_camera.json
   Reports reprojection RMS on the views used AND on held-out views.
3. Plane: one photo with the target lying flat AT BOARD-TOP HEIGHT (a 1.6 mm
   board under the target, or the target taped on a bare board), in the
   camera's working position.
       python -m metrology.charuco_calib homography plane.jpg --intrinsics top_camera.json --out plane.json
   plane.json holds the POST body for /calibration/save, intrinsics included.

`demo` runs steps 2-3 on rendered images of a made-up camera and says so.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DICTIONARIES = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
                "DICT_6X6_250": cv2.aruco.DICT_6X6_250}


def make_board(squares=(9, 6), square_mm=10.0, marker_mm=7.5, dictionary="DICT_5X5_100"):
    d = cv2.aruco.getPredefinedDictionary(DICTIONARIES[dictionary])
    return cv2.aruco.CharucoBoard(tuple(squares), float(square_mm), float(marker_mm), d)


def board_size_mm(board):
    sx, sy = board.getChessboardSize()
    return sx * board.getSquareLength(), sy * board.getSquareLength()


def board_image(board, px_per_mm):
    w, h = board_size_mm(board)
    return board.generateImage((int(round(w * px_per_mm)), int(round(h * px_per_mm))), marginSize=0, borderBits=1)


def detect(board, image, min_corners=6):
    """-> (object points (N,3) mm, image points (N,2) px, ids) or None."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32), ids.ravel()


def _view_rms(obj, img, K, dist):
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
    if not ok:
        return float("inf")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def calibrate_intrinsics(board, images, holdout_every=5, flags=0):
    """Calibrate on all views except every `holdout_every`-th, then check the
    held-out views with the fitted intrinsics (pose solved per view). A large
    gap between the two RMS values means the fit doesn't generalise: more
    views, more tilt, more coverage of the image corners."""
    views, size = [], None
    for im in images:
        det = detect(board, im)
        size = (im.shape[1], im.shape[0])
        if det is not None:
            views.append(det)
    if len(views) < 4:
        raise ValueError("only %d usable views; need at least 4 (15-30 recommended)" % len(views))
    held = [v for i, v in enumerate(views) if holdout_every and i % holdout_every == holdout_every - 1]
    used = [v for i, v in enumerate(views) if not (holdout_every and i % holdout_every == holdout_every - 1)]
    rms, K, dist, _, _ = cv2.calibrateCamera([v[0] for v in used], [v[1] for v in used], size, None, None, flags=flags)
    per_view = [_view_rms(o, p, K, dist) for o, p, _ in used]
    held_rms = [_view_rms(o, p, K, dist) for o, p, _ in held]
    return {
        "image_size": list(size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms_px": float(rms),
        "heldout_rms_px": float(np.sqrt(np.mean(np.square(held_rms)))) if held_rms else None,
        "worst_view_rms_px": float(max(per_view)),
        "views_used": len(used),
        "views_heldout": len(held),
        "views_rejected": len(images) - len(views),
        "board": {"squares": list(board.getChessboardSize()), "square_mm": board.getSquareLength(),
                  "marker_mm": board.getMarkerLength()},
    }


def undistort_pixels(points, camera_matrix, dist_coeffs):
    """Distorted pixel coordinates -> where an ideal (distortion-free) camera
    with the same K would have seen them, still in pixels."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(camera_matrix, dtype=np.float64)
    out = cv2.undistortPointsIter(pts, K, np.asarray(dist_coeffs, dtype=np.float64), None, K,
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9))
    return out.reshape(-1, 2)


def homography_residuals(pixel_points, world_points):
    """Fit H on every other point, test on the rest, then fit on all.
    The held-out number is the honest one: with exactly 4 points the
    in-sample residual is zero by construction."""
    src = np.asarray(pixel_points, dtype=np.float64)
    dst = np.asarray(world_points, dtype=np.float64)
    for pts in (src, dst):
        # OpenCV returns a degenerate H for collinear points instead of failing
        sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        if sv[-1] <= 1e-6 * max(sv[0], 1e-12):
            raise ValueError("Homography computation failed: the points are collinear")
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None or not np.all(np.isfinite(H)):
        raise ValueError("Homography computation failed (collinear points?)")
    fit = np.linalg.norm(cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2) - dst, axis=1)
    out = {"homography": H.tolist(), "n_points": len(src),
           "rms_mm": float(np.sqrt(np.mean(fit ** 2))), "max_mm": float(fit.max()), "heldout_rms_mm": None}
    if len(src) >= 10:
        a, b = np.arange(0, len(src), 2), np.arange(1, len(src), 2)
        Ha, _ = cv2.findHomography(src[a], dst[a], 0)
        if Ha is not None:
            e = np.linalg.norm(cv2.perspectiveTransform(src[b].reshape(-1, 1, 2), Ha).reshape(-1, 2) - dst[b], axis=1)
            out["heldout_rms_mm"] = float(np.sqrt(np.mean(e ** 2)))
    return out


def plane_homography(board, image, intrinsics, y_up=True):
    """Homography from undistorted pixels to the target plane in mm. With
    y_up the origin is the target's bottom-left corner as printed, x right,
    y up: right-handed, like a PCB frame (kinematic_fit expects that)."""
    det = detect(board, image)
    if det is None:
        raise ValueError("ChArUco target not found in the plane image")
    obj, img, _ = det
    world = obj[:, :2].astype(np.float64)
    if y_up:
        world[:, 1] = board_size_mm(board)[1] - world[:, 1]
    K, dist = intrinsics["camera_matrix"], intrinsics["dist_coeffs"]
    corrected = homography_residuals(undistort_pixels(img, K, dist), world)
    raw = homography_residuals(img, world)
    return {
        "corrected": corrected,
        "without_distortion_correction": {k: raw[k] for k in ("rms_mm", "max_mm", "heldout_rms_mm")},
        "save_request": {   # POST body for /calibration/save
            "pixel_points": img.astype(float).tolist(),
            "world_points": world.tolist(),
            "intrinsics": {"camera_matrix": K, "dist_coeffs": list(dist)},
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# Synthetic views, for tests and the demo
# ─────────────────────────────────────────────────────────────────────────

def render_view(board_img, px_per_mm, K, dist, rvec, tvec, size, noise=2.0, seed=0):
    """Image of the flat target as a camera with intrinsics K and distortion
    `dist` would see it at pose (rvec, tvec): every output pixel is
    undistorted, cast as a ray and intersected with the target plane."""
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    norm = cv2.undistortPointsIter(pix, K, dist, None, None,
                                   (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-9)).reshape(-1, 2)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    Hn = np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, dtype=np.float64).ravel()])
    plane = np.linalg.solve(Hn, np.column_stack([norm, np.ones(len(norm))]).T).T
    X, Y = plane[:, 0] / plane[:, 2], plane[:, 1] / plane[:, 2]
    behind = plane[:, 2] <= 0
    mx = (X * px_per_mm).reshape(h, w).astype(np.float32)
    my = (Y * px_per_mm).reshape(h, w).astype(np.float32)
    mx[behind.reshape(h, w)] = -1e6
    out = cv2.remap(board_img, mx, my, cv2.INTER_AREA if px_per_mm > 4 else cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    out = cv2.GaussianBlur(out, (3, 3), 0.7)
    if noise:
        out = np.clip(out + np.random.default_rng(seed).normal(0, noise, out.shape), 0, 255)
    return out.astype(np.uint8)


DEMO_K = np.array([[820.0, 0.0, 322.0], [0.0, 818.0, 236.0], [0.0, 0.0, 1.0]])
DEMO_DIST = np.array([-0.21, 0.09, 0.0006, -0.0004, 0.0])


def demo_views(board, n=15, size=(640, 480), seed=0):
    """Rendered views of a made-up camera (DEMO_K, DEMO_DIST) looking at the
    target from varied tilts and offsets, so the image corners are covered."""
    rng = np.random.default_rng(seed)
    img = board_image(board, 10.0)
    bw, bh = board_size_mm(board)
    views = []
    for i in range(n):
        rx, ry = rng.uniform(-0.6, 0.6, 2)
        rz = rng.uniform(-0.3, 0.3)
        z = rng.uniform(260, 340)
        # place the board centre somewhere in the image, edges included
        cx, cy = rng.uniform(-0.25, 0.25) * z, rng.uniform(-0.2, 0.2) * z
        R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
        t = np.array([cx, cy, z]) - R @ np.array([bw / 2, bh / 2, 0.0])
        views.append(render_view(img, 10.0, DEMO_K, DEMO_DIST, cv2.Rodrigues(R)[0], t, size, seed=seed + i))
    return views


def demo_plane_view(board, size=(640, 480)):
    """The target lying flat under a slightly tilted camera, filling the view."""
    bw, bh = board_size_mm(board)
    R, _ = cv2.Rodrigues(np.array([0.12, -0.08, 0.03]))
    t = np.array([0.0, 0.0, 120.0]) - R @ np.array([bw / 2, bh / 2, 0.0])
    return render_view(board_image(board, 10.0), 10.0, DEMO_K, DEMO_DIST, cv2.Rodrigues(R)[0], t, size, seed=99)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def _board_from_args(a):
    return make_board((a.squares_x, a.squares_y), a.square_mm, a.marker_mm, a.dictionary)


def _load_images(folder):
    paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif") for p in glob.glob(os.path.join(folder, ext)))
    return paths, [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]


def main(argv=None):
    ap = argparse.ArgumentParser(description="ChArUco camera calibration and board-plane homography.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make-board", "intrinsics", "homography", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--squares-x", type=int, default=9)
        p.add_argument("--squares-y", type=int, default=6)
        p.add_argument("--square-mm", type=float, default=10.0, help="MEASURED square size of your print")
        p.add_argument("--marker-mm", type=float, default=7.5)
        p.add_argument("--dictionary", default="DICT_5X5_100", choices=sorted(DICTIONARIES))
        if name == "make-board":
            p.add_argument("--out", required=True)
            p.add_argument("--dpi", type=int, default=600)
        elif name == "intrinsics":
            p.add_argument("folder")
            p.add_argument("--out", required=True)
        elif name == "homography":
            p.add_argument("image")
            p.add_argument("--intrinsics", required=True)
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    board = _board_from_args(a)

    if a.cmd == "make-board":
        from PIL import Image
        img = board_image(board, a.dpi / 25.4)
        Image.fromarray(img).save(a.out, dpi=(a.dpi, a.dpi))
        w, h = board_size_mm(board)
        print("Wrote %s: %.0f x %.0f mm at %d dpi. Print at 100%% (no 'fit to page'), glue flat to something rigid, "
              "then measure a square with calipers and pass that as --square-mm." % (a.out, w, h, a.dpi))
        return 0

    if a.cmd == "intrinsics":
        paths, images = _load_images(a.folder)
        res = calibrate_intrinsics(board, [im for im in images if im is not None])
        res["source_folder"] = str(Path(a.folder).resolve())
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("Views used %d, held out %d, rejected %d. Reprojection RMS %.3f px, held-out %s px, worst view %.3f px."
              % (res["views_used"], res["views_heldout"], res["views_rejected"], res["rms_px"],
                 "%.3f" % res["heldout_rms_px"] if res["heldout_rms_px"] is not None else "n/a", res["worst_view_rms_px"]))
        print("Wrote %s" % a.out)
        return 0

    if a.cmd == "homography":
        intr = json.loads(Path(a.intrinsics).read_text(encoding="utf-8"))
        res = plane_homography(board, cv2.imread(a.image, cv2.IMREAD_GRAYSCALE), intr)
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        _print_plane(res)
        print("Wrote %s (save_request is the POST body for /calibration/save)" % a.out)
        return 0

    # demo
    print("SYNTHETIC: rendered views of a made-up camera, not yours.")
    res = calibrate_intrinsics(board, demo_views(board))
    K = np.array(res["camera_matrix"])
    print("Reprojection RMS %.3f px, held-out %.3f px." % (res["rms_px"], res["heldout_rms_px"]))
    print("fx %.1f (true %.1f), fy %.1f (true %.1f), k1 %.3f (true %.3f)"
          % (K[0, 0], DEMO_K[0, 0], K[1, 1], DEMO_K[1, 1], res["dist_coeffs"][0], DEMO_DIST[0]))
    _print_plane(plane_homography(board, demo_plane_view(board), res))
    return 0


def _print_plane(res):
    c, r = res["corrected"], res["without_distortion_correction"]
    fmt = lambda v: "n/a" if v is None else "%.4f" % v
    print("Plane homography on %d points: RMS %s mm, held-out %s mm, max %s mm"
          % (c["n_points"], fmt(c["rms_mm"]), fmt(c["heldout_rms_mm"]), fmt(c["max_mm"])))
    print("Same points without distortion correction: RMS %s mm, held-out %s mm, max %s mm"
          % (fmt(r["rms_mm"]), fmt(r["heldout_rms_mm"]), fmt(r["max_mm"])))


if __name__ == "__main__":
    sys.exit(main())
