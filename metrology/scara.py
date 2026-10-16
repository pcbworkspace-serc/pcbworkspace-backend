"""
scara.py — planar 2-link (SCARA XY) kinematics for the MiniMEE arm, and a
least-squares fit of its real geometry from measured tip positions.

Frames. Board coordinates are mm in the board frame (what the top-camera
homography and a KiCad .pos file give). The arm frame has its origin on the
base joint axis and its axes parallel to the board's, so

    p_arm = p_board + base_offset      (same convention as layout_engine.ARM_BASE_OFFSET_MM)

A rotation between the two frames is absorbed by the base joint's zero
offset: from tip measurements alone the two can't be told apart, so there
is deliberately no separate frame-rotation parameter.

Joints. theta_i = scale_i * q_i + zero_i, where q_i is the commanded joint
angle in degrees (what the firmware is told) and theta_i the true one.
scale_i catches a wrong steps-per-degree; zero_i is where the joint really
is when the firmware thinks it's at 0.

No scipy: the fit is a small Levenberg-Marquardt over 6-8 parameters.
"""

import math
from dataclasses import dataclass

import numpy as np

PARAM_NAMES = ("base_x_mm", "base_y_mm", "l1_mm", "l2_mm", "zero1_deg", "zero2_deg", "scale1", "scale2")


@dataclass
class ArmGeometry:
    base_offset_mm: tuple = (190.0, 40.0)
    l1_mm: float = 150.0
    l2_mm: float = 180.0
    zero_deg: tuple = (0.0, 0.0)
    scale: tuple = (1.0, 1.0)

    def to_vector(self):
        return np.array([*self.base_offset_mm, self.l1_mm, self.l2_mm, *self.zero_deg, *self.scale], dtype=float)

    @classmethod
    def from_vector(cls, v):
        v = [float(x) for x in v]
        return cls((v[0], v[1]), v[2], v[3], (v[4], v[5]), (v[6], v[7]))

    def to_dict(self):
        return {"base_offset_mm": list(self.base_offset_mm), "l1_mm": self.l1_mm, "l2_mm": self.l2_mm,
                "zero_deg": list(self.zero_deg), "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["base_offset_mm"]), float(d["l1_mm"]), float(d["l2_mm"]),
                   tuple(d.get("zero_deg", (0.0, 0.0))), tuple(d.get("scale", (1.0, 1.0))))


def _true_angles_rad(geom, q_deg):
    q = np.atleast_2d(np.asarray(q_deg, dtype=float))
    t1 = np.radians(geom.scale[0] * q[:, 0] + geom.zero_deg[0])
    t2 = np.radians(geom.scale[1] * q[:, 1] + geom.zero_deg[1])
    return t1, t2


def forward(geom, q_deg):
    """Commanded joint angles (N,2) in degrees -> tip position (N,2) in board mm."""
    t1, t2 = _true_angles_rad(geom, q_deg)
    x = geom.l1_mm * np.cos(t1) + geom.l2_mm * np.cos(t1 + t2)
    y = geom.l1_mm * np.sin(t1) + geom.l2_mm * np.sin(t1 + t2)
    return np.column_stack([x - geom.base_offset_mm[0], y - geom.base_offset_mm[1]])


def inverse(geom, xy_board, elbow=1):
    """Board positions (N,2) mm -> commanded joint angles (N,2) degrees.
    elbow=+1 picks theta2 > 0, -1 the mirror solution. Unreachable points
    come back as NaN rather than raising, so a whole grid can be mapped."""
    p = np.atleast_2d(np.asarray(xy_board, dtype=float)) + np.asarray(geom.base_offset_mm)
    l1, l2 = geom.l1_mm, geom.l2_mm
    c2 = (p[:, 0] ** 2 + p[:, 1] ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    ok = np.abs(c2) <= 1.0
    t2 = elbow * np.arccos(np.clip(c2, -1.0, 1.0))
    t1 = np.arctan2(p[:, 1], p[:, 0]) - np.arctan2(l2 * np.sin(t2), l1 + l2 * np.cos(t2))
    q1 = (np.degrees(t1) - geom.zero_deg[0]) / geom.scale[0]
    q2 = (np.degrees(t2) - geom.zero_deg[1]) / geom.scale[1]
    q1 = (q1 + 180.0) % 360.0 - 180.0
    q = np.column_stack([q1, q2])
    q[~ok] = np.nan
    return q


def local_directions(geom, q_deg):
    """Per pose, the tip-motion vectors the error budget needs, all in the
    board/arm plane: d(tip)/d(theta1) and d(tip)/d(theta2) in mm per radian,
    and unit vectors along link 1, link 2 and base->tip. Shapes (N,2)."""
    t1, t2 = _true_angles_rad(geom, q_deg)
    l1, l2 = geom.l1_mm, geom.l2_mm
    elbow = np.column_stack([l1 * np.cos(t1), l1 * np.sin(t1)])
    link2 = np.column_stack([l2 * np.cos(t1 + t2), l2 * np.sin(t1 + t2)])
    tip = elbow + link2
    perp = lambda v: np.column_stack([-v[:, 1], v[:, 0]])
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    return {"d_theta1": perp(tip), "d_theta2": perp(link2), "u_link1": unit(elbow),
            "u_link2": unit(link2), "u_radial": unit(tip), "r_mm": np.linalg.norm(tip, axis=1)}


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def canonical(geom):
    """The same physical arm can be written with a negative link length and
    a zero offset 180 degrees away (and zeros 360 apart). Pick the one
    with positive links and zeros in (-180, 180] so fits are comparable."""
    l1, l2 = geom.l1_mm, geom.l2_mm
    z1, z2 = geom.zero_deg
    if l1 < 0:
        l1, z1, z2 = -l1, z1 + 180.0, z2 - 180.0
    if l2 < 0:
        l2, z2 = -l2, z2 + 180.0
    return ArmGeometry(geom.base_offset_mm, l1, l2, (_wrap_deg(z1), _wrap_deg(z2)), geom.scale)


# ─────────────────────────────────────────────────────────────────────────
# Least-squares fit
# ─────────────────────────────────────────────────────────────────────────

def _numeric_jacobian(fun, x, r0):
    J = np.empty((r0.size, x.size))
    for i in range(x.size):
        h = 1e-6 * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        J[:, i] = (fun(xp) - fun(xm)) / (2 * h)
    return J


def _levenberg_marquardt(fun, x0, max_iter=200):
    x = np.asarray(x0, dtype=float).copy()
    r = fun(x)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(max_iter):
        J = _numeric_jacobian(fun, x, r)
        A, g = J.T @ J, J.T @ r
        improved = False
        while lam < 1e12:
            dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            r_new = fun(x + dx)
            c_new = float(r_new @ r_new)
            if c_new < cost:
                x, r, cost, improved = x + dx, r_new, c_new, True
                lam = max(lam / 10.0, 1e-12)
                break
            lam *= 10.0
        if not improved or np.linalg.norm(dx) < 1e-12 * (1.0 + np.linalg.norm(x)):
            break
    return x, r, _numeric_jacobian(fun, x, r)


@dataclass
class FitResult:
    geometry: ArmGeometry
    fitted: tuple                # names of the parameters that were free
    std_error: dict              # 1-sigma standard error per free parameter
    covariance: np.ndarray       # covariance of the free parameters, same order as `fitted`
    max_abs_correlation: float   # > 0.95 means two parameters are nearly indistinguishable
    residuals_mm: np.ndarray     # (N,2) measured - model
    rms_mm: float                # sqrt(mean |residual|^2)
    max_mm: float
    cv_rms_mm: float             # k-fold held-out RMS: what a NEW pose should expect
    n_points: int
    mirrored_fits_better: bool   # board y axis probably flipped vs the arm (e.g. image coordinates)

    def summary(self):
        return {
            "geometry": self.geometry.to_dict(),
            "fitted_parameters": list(self.fitted),
            "std_error": self.std_error,
            "covariance": [[float(v) for v in row] for row in self.covariance],
            "max_abs_correlation": round(self.max_abs_correlation, 4),
            "residual_rms_mm": round(self.rms_mm, 4),
            "residual_max_mm": round(self.max_mm, 4),
            "heldout_rms_mm": None if math.isnan(self.cv_rms_mm) else round(self.cv_rms_mm, 4),
            "n_points": self.n_points,
            "mirrored_fits_better": self.mirrored_fits_better,
        }


def _fit_once(q_deg, xy, initial, free, multistart=True):
    full0 = initial.to_vector()

    def unpack(x):
        v = full0.copy()
        v[free] = x
        return ArmGeometry.from_vector(v)

    def residual(x):
        return (forward(unpack(x), q_deg) - xy).ravel()

    best = None
    # The base zero offset can be anywhere on the circle if the arm is
    # mounted rotated relative to the board; LM from a single start can
    # settle in the wrong basin, so seed it every 30 degrees.
    starts = [0.0] if (4 not in free or not multistart) else [((s + 180.0) % 360.0) - 180.0 for s in range(0, 360, 30)]
    for s in starts:
        x0 = full0[free].copy()
        if 4 in free:
            x0[list(free).index(4)] = initial.zero_deg[0] + s
        x, r, J = _levenberg_marquardt(residual, x0)
        cost = float(r @ r)
        if best is None or cost < best[0]:
            best = (cost, x, r, J)
    cost, x, r, J = best
    return canonical(unpack(x)), r.reshape(-1, 2), J, cost


def fit_geometry(q_deg, xy_meas, initial=None, fit_scale=None, folds=5):
    """Fit the arm's geometry to measured tip positions.

    q_deg:   (N,2) commanded joint angles, degrees
    xy_meas: (N,2) measured tip positions, board mm
    fit_scale: also fit steps-per-degree errors. Default: only when there are
               at least 16 points and each joint was moved over 30+ degrees,
               otherwise the scale terms are too weakly determined to trust.
    """
    q = np.asarray(q_deg, dtype=float)
    xy = np.asarray(xy_meas, dtype=float)
    if q.shape != xy.shape or q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("q_deg and xy_meas must both be (N,2)")
    initial = initial or ArmGeometry()
    if fit_scale is None:
        spans = np.ptp(q, axis=0)
        fit_scale = len(q) >= 16 and bool(np.all(spans >= 30.0))
    free = np.arange(8) if fit_scale else np.arange(6)
    if len(q) < len(free) + 2:
        raise ValueError("need at least %d measured poses to fit %d parameters with any redundancy"
                         % (len(free) + 2, len(free)))

    geom, res, J, cost = _fit_once(q, xy, initial, free)

    # Same data with the board's y axis flipped: if that fits far better,
    # the measurements were almost certainly taken in image coordinates
    # (y down) or a mirrored CAD frame. The arm model is right-handed.
    # With the scales free, a mirror instead fits perfectly with BOTH
    # scales near -1 (a y flip is the same as reversing both joints), so
    # that counts too. One negative scale means one motor turns backwards.
    _, _, _, cost_m = _fit_once(q, xy * np.array([1.0, -1.0]), initial, free)
    mirrored = (cost_m < 0.25 * cost and cost > 1e-6) or all(s < 0 for s in geom.scale)

    dof = max(1, res.size - len(free))
    sigma2 = cost / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        corr = cov / np.outer(d, d)
        np.fill_diagonal(corr, 0.0)
        max_corr = float(np.max(np.abs(corr))) if corr.size > 1 else 0.0
    except np.linalg.LinAlgError:
        cov = np.full((len(free), len(free)), np.inf)
        se = np.full(len(free), np.inf)
        max_corr = 1.0

    radial = np.linalg.norm(res, axis=1)

    cv_rms = float("nan")
    k = min(folds, len(q))
    if k >= 3 and len(q) - math.ceil(len(q) / k) >= len(free) + 2:
        order = np.random.default_rng(0).permutation(len(q))
        errs = []
        for f in range(k):
            test = order[f::k]
            train = np.setdiff1d(order, test)
            g, _, _, _ = _fit_once(q[train], xy[train], geom, free, multistart=False)
            errs.append(np.linalg.norm(forward(g, q[test]) - xy[test], axis=1))
        cv_rms = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))

    return FitResult(
        geometry=geom,
        fitted=tuple(PARAM_NAMES[i] for i in free),
        std_error={PARAM_NAMES[i]: float(s) for i, s in zip(free, se)},
        covariance=cov,
        max_abs_correlation=max_corr,
        residuals_mm=res,
        rms_mm=float(np.sqrt(np.mean(radial ** 2))),
        max_mm=float(np.max(radial)),
        cv_rms_mm=cv_rms,
        n_points=len(q),
        mirrored_fits_better=bool(mirrored),
    )


def tip_covariance(geom, q_deg, names, cov):
    """Propagate a fit's parameter covariance to the tip at each pose:
    G C G^T with G = d(tip)/d(params). This keeps the correlations (a base
    offset error and a zero offset error partly cancel inside the fitted
    region), which adding per-parameter errors independently would not."""
    q = np.atleast_2d(np.asarray(q_deg, dtype=float))
    v0 = geom.to_vector()
    cols = []
    for name in names:
        k = PARAM_NAMES.index(name)
        h = 1e-6 * max(1.0, abs(v0[k]))
        vp, vm = v0.copy(), v0.copy()
        vp[k] += h
        vm[k] -= h
        cols.append((forward(ArmGeometry.from_vector(vp), q) - forward(ArmGeometry.from_vector(vm), q)) / (2 * h))
    G = np.stack(cols, axis=-1)                                   # (N,2,p)
    return np.einsum("nip,pq,njq->nij", G, np.asarray(cov, dtype=float), G)


def predicted_fit_covariance(geom, q_deg, sigma_mm, fit_scale=None):
    """Parameter covariance a fit WOULD have if these poses are measured with
    per-axis noise sigma_mm: sigma^2 (J^T J)^-1. Uses the same rule as
    fit_geometry for whether the joint scales are fitted."""
    q = np.asarray(q_deg, dtype=float)
    if fit_scale is None:
        fit_scale = len(q) >= 16 and bool(np.all(np.ptp(q, axis=0) >= 30.0))
    free = np.arange(8) if fit_scale else np.arange(6)
    v0 = geom.to_vector()

    def tip(x):
        v = v0.copy()
        v[free] = x
        return forward(ArmGeometry.from_vector(v), q).ravel()

    x0 = v0[free]
    J = _numeric_jacobian(tip, x0, tip(x0))
    return [PARAM_NAMES[i] for i in free], sigma_mm ** 2 * np.linalg.inv(J.T @ J)


def plan_calibration_poses(geom, board_mm, n=25, margin_mm=15.0, elbow=1, seed=0):
    """Spread n target poses over the board area plus a margin (more spread =
    better-determined link lengths and zeros), keeping only reachable ones.
    Returns (xy_board (n,2), q_deg (n,2)) from the NOMINAL geometry: these are
    commands to send, the measured positions are what the fit needs."""
    w, h = board_mm
    side = int(math.ceil(math.sqrt(n)))
    xs = np.linspace(-margin_mm, w + margin_mm, side)
    ys = np.linspace(-margin_mm, h + margin_mm, side)
    grid = np.array([(x, y) for y in ys for x in xs])
    q = inverse(geom, grid, elbow=elbow)
    keep = ~np.isnan(q).any(axis=1)
    grid, q = grid[keep], q[keep]
    if len(grid) > n:
        idx = np.sort(np.random.default_rng(seed).choice(len(grid), n, replace=False))
        grid, q = grid[idx], q[idx]
    return grid, q
