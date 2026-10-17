"""Tests for vision_eval.py: metric math, CI coverage, the provenance gate
and the board/session leakage guard.

Every capture written here is a SYNTHETIC test fixture: drawn rectangles on
noise, labeled by the test itself, saved through data_capture.py with
capture_context.synthetic = True so the harness stamps any report built from
them SYNTHETIC_TEST_FIXTURE. None of these numbers describe a SERC board.
"""
import json
import math

import cv2
import numpy as np
import pytest
from PIL import Image

import data_capture as dc
import vision_eval as ve

SYNTHETIC = {"synthetic": True, "note": "SYNTHETIC test fixture, not a SERC capture"}


# ─────────────────────────────────────────────────────────────────────────
# Synthetic fixture helpers
# ─────────────────────────────────────────────────────────────────────────

def _rect_image(seed, center=(80, 60), wh=(50, 24), angle=0.0, size=(160, 120)):
    """Dark rectangle on light noise. The noise seed makes every frame's
    pixels unique, so no two fixtures trip the identical-frame guard."""
    rng = np.random.default_rng(seed)
    canvas = rng.integers(205, 256, size=(size[1], size[0]), dtype=np.uint8)
    box = cv2.boxPoints((center, wh, angle)).astype(np.int32)
    cv2.fillPoly(canvas, [box], 25)
    return Image.fromarray(canvas).convert("RGB")


def _add(base, kind, img, gt, board=None, session=None, prediction=None,
         label_source="human", labeler="fixture-labeler", px_per_mm=None, synthetic=True):
    ctx = dict(SYNTHETIC) if synthetic else {}
    if px_per_mm:
        ctx["px_per_mm"] = px_per_mm
    cid = dc.save_capture(kind, img, prediction or {}, base_dir=str(base),
                          board_id=board, session_id=session, context=ctx)
    dc.record_feedback(kind, cid, gt, base_dir=str(base), label_source=label_source,
                       labeler=labeler if label_source == "human" else None)
    return cid


def _cfg(**kw):
    kw.setdefault("test_fraction", 1.0)  # every group held out unless a test says otherwise
    kw.setdefault("n_boot", 400)
    return ve.EvalConfig(**kw)


def _lookup(table):
    """A fake model keyed by frame content, so the test fixes its outputs."""
    return lambda img: table[ve._pixel_digest(img.convert("RGB"))]


def _board_on_side(side, cfg, prefix="b"):
    for i in range(10_000):
        name = f"{prefix}{i}"
        in_test = ve._unit_hash(cfg.split_seed, "board:" + name) < cfg.test_fraction
        if in_test == (side == "test"):
            return name
    raise AssertionError("no board name found")


# ─────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────

def test_wilson_matches_reference_values():
    lo, hi = ve.wilson_interval(8, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.4902, 0.9433)
    lo, hi = ve.wilson_interval(0, 10)
    assert lo == 0.0 and round(hi, 4) == 0.2775
    lo, hi = ve.wilson_interval(10, 10)
    assert hi == 1.0 and round(lo, 4) == 0.7225
    assert ve.wilson_interval(0, 0) is None


def test_wilson_coverage_is_near_nominal():
    rng = np.random.default_rng(1)
    for p, n in ((0.3, 40), (0.9, 60), (0.5, 15)):
        draws = rng.binomial(n, p, size=4000)
        covered = [lo <= p <= hi for lo, hi in (ve.wilson_interval(int(k), n) for k in draws)]
        assert 0.92 <= np.mean(covered) <= 0.98, (p, n, np.mean(covered))


def _clustered_sample(rng, n_groups=30, per_group=10, mu=2.0):
    effects = rng.normal(0.0, 1.0, size=n_groups)
    values = np.concatenate([mu + e + rng.normal(0.0, 0.3, size=per_group) for e in effects])
    groups = [f"g{i}" for i in range(n_groups) for _ in range(per_group)]
    return values, groups


def test_board_bootstrap_covers_clustered_mean_where_frame_bootstrap_does_not():
    """Frames on one board share that board's error. Resampling boards keeps
    near-nominal coverage of the true mean; resampling frames as if they
    were independent gives far too narrow an interval. This is why the
    harness resamples boards/sessions."""
    rng = np.random.default_rng(7)
    board_cfg = ve.EvalConfig(n_boot=500, min_groups_for_board_bootstrap=5)
    frame_cfg = ve.EvalConfig(n_boot=500, min_groups_for_board_bootstrap=10 ** 9)
    hits = {"board": 0, "frame": 0}
    reps = 200
    for _ in range(reps):
        values, groups = _clustered_sample(rng)
        for name, cfg in (("board", board_cfg), ("frame", frame_cfg)):
            res = ve.Resampler(groups, cfg, rng)
            lo, hi = res.ci((res.weights @ values) / res.weights.sum(axis=1))
            hits[name] += lo <= 2.0 <= hi
    assert hits["board"] / reps >= 0.88
    assert hits["frame"] / reps <= 0.6


def test_summary_quantiles_match_numpy_inverted_cdf():
    v = np.random.default_rng(3).exponential(1.0, size=37)
    groups = [f"g{i % 7}" for i in range(v.size)]
    res = ve.Resampler(groups, ve.EvalConfig(n_boot=200), np.random.default_rng(0))
    out = ve.summarize_errors(v, groups, res)
    assert out["mean"]["value"] == round(v.mean(), 4)
    assert out["median"]["value"] == round(float(np.quantile(v, 0.5, method="inverted_cdf")), 4)
    assert out["p95"]["value"] == round(float(np.quantile(v, 0.95, method="inverted_cdf")), 4)
    for stat in ("mean", "median", "p95"):
        lo, hi = out[stat]["ci95"]
        assert lo <= out[stat]["value"] <= hi
        assert out[stat]["ci_method"] == "board_bootstrap"


# ─────────────────────────────────────────────────────────────────────────
# Matching and geometry
# ─────────────────────────────────────────────────────────────────────────

def test_match_frame_counts_tp_fp_fn_per_class():
    gt = [{"class": "R", "box": [0, 0, 10, 10]},
          {"class": "R", "box": [20, 0, 30, 10]},
          {"class": "C", "box": [0, 20, 10, 30]}]
    preds = [{"class": "R", "box": [0, 0, 10, 10], "score": 0.9},     # TP
             {"class": "R", "box": [1, 0, 11, 10], "score": 0.8},     # duplicate of a taken box: FP
             {"class": "C", "box": [5, 20, 15, 30], "score": 0.7},    # IoU 1/3 < 0.5: FP, and C is missed
             {"class": "LED", "box": [50, 50, 60, 60], "score": 0.6}]  # nothing there: FP
    assert ve.match_frame(gt, preds, 0.5) == {"R": [1, 1, 1], "C": [0, 1, 1], "LED": [0, 1, 0]}


def test_match_frame_is_class_aware_and_prefers_higher_score():
    gt = [{"class": "R", "box": [0, 0, 10, 10]}]
    assert ve.match_frame(gt, [{"class": "C", "box": [0, 0, 10, 10], "score": 1}], 0.5) == {
        "R": [0, 0, 1], "C": [0, 1, 0]}
    preds = [{"class": "R", "box": [0, 0, 10, 9], "score": 0.4},
             {"class": "R", "box": [0, 0, 10, 10], "score": 0.9}]
    assert ve.match_frame(gt, preds, 0.5) == {"R": [1, 1, 0]}


def test_angle_error_wraps_by_period():
    assert ve.angle_error_deg(179, -179) == pytest.approx(-2)
    assert ve.angle_error_deg(44, -44, period=90) == pytest.approx(-2)
    assert abs(ve.angle_error_deg(10, 190)) == pytest.approx(180)
    assert ve.angle_error_deg(10, 190, period=180) == pytest.approx(0)


# ─────────────────────────────────────────────────────────────────────────
# Provenance gate: only a person's labels get reported
# ─────────────────────────────────────────────────────────────────────────

def _one_align_frame(base, **kw):
    return _add(base, "align", _rect_image(1), {"delta_theta_deg": 3.0, "delta_x_mm": 0.1, "delta_y_mm": 0.0},
                board="B1", **kw)


def test_refuses_labels_marked_as_model(tmp_path):
    cid = _one_align_frame(tmp_path, label_source="model")
    with pytest.raises(ve.EvalRefused) as e:
        ve.evaluate(tmp_path, aligners={"x": lambda img: {}}, cfg=_cfg())
    assert any(cid in p and "'model'" in p for p in e.value.problems)


def test_refuses_labels_with_no_recorded_source(tmp_path):
    cid = dc.save_capture("align", _rect_image(2), {}, base_dir=str(tmp_path), board_id="B1", context=SYNTHETIC)
    dc.record_feedback("align", cid, {"delta_theta_deg": 1.0}, base_dir=str(tmp_path))  # legacy: no label_source
    with pytest.raises(ve.EvalRefused) as e:
        ve.evaluate(tmp_path, aligners={"x": lambda img: {}}, cfg=_cfg())
    assert "unspecified" in e.value.problems[0]


def test_refuses_human_label_identical_to_model_prediction(tmp_path):
    pred = {"available": True, "delta_theta_deg": 4.12, "delta_x_mm": 0.031, "delta_y_mm": -0.2}
    _add(tmp_path, "align", _rect_image(3), {k: pred[k] for k in ("delta_theta_deg", "delta_x_mm", "delta_y_mm")},
         board="B1", prediction=pred)
    with pytest.raises(ve.EvalRefused) as e:
        ve.evaluate(tmp_path, aligners={"x": lambda img: {}}, cfg=_cfg())
    assert "identical to the model's own prediction" in e.value.problems[0]


def test_reserved_split_labels_are_not_held_to_the_human_rule(tmp_path):
    """Pseudo-labels on the fine-tuning side are legitimate; only the frames
    that get reported must be a person's."""
    cfg = _cfg(test_fraction=0.5)
    _add(tmp_path, "align", _rect_image(4), {"delta_theta_deg": 0.0}, board=_board_on_side("train", cfg),
         label_source="model")
    _add(tmp_path, "align", _rect_image(5), {"delta_theta_deg": 0.0}, board=_board_on_side("test", cfg))
    report = ve.evaluate(tmp_path, aligners={"x": lambda img: {"available": True, "delta_theta_deg": 0.0}}, cfg=cfg)
    assert report["alignment"]["n_frames"] == 1


def test_refuses_malformed_ground_truth(tmp_path):
    _add(tmp_path, "detect", _rect_image(6), {"boxes": [{"class": "R", "box": [10, 10, 5, 20]}]}, board="B1")
    with pytest.raises(ve.EvalRefused) as e:
        ve.evaluate(tmp_path, detectors={"d": lambda img: []}, cfg=_cfg())
    assert "x2>x1" in e.value.problems[0]


def test_refuses_mixed_synthetic_and_real_frames(tmp_path):
    _add(tmp_path, "align", _rect_image(7), {"delta_theta_deg": 0.0}, board="B1")
    _add(tmp_path, "align", _rect_image(8), {"delta_theta_deg": 0.0}, board="B2", synthetic=False)
    with pytest.raises(ve.EvalRefused, match="mixed"):
        ve.evaluate(tmp_path, cfg=_cfg())


# ─────────────────────────────────────────────────────────────────────────
# Leakage guard: split by board/session, never by frame
# ─────────────────────────────────────────────────────────────────────────

def _rec(key, board=None, session=None):
    return {"_key": key, "capture_id": key, "board_id": board, "session_id": session}


def test_boards_and_sessions_that_share_frames_form_one_group():
    records = [_rec("a", "B1", "S1"), _rec("b", "B1", "S2"), _rec("c", "B2", "S2"),
               _rec("d", "B3", "S3"), _rec("e")]
    group_of, members, ungrouped = ve.group_records(records)
    assert group_of["a"] == group_of["b"] == group_of["c"]
    assert group_of["d"] != group_of["a"]
    assert members[group_of["a"]] == {"board:B1", "board:B2", "session:S1", "session:S2"}
    assert ungrouped == ["e"]


def test_split_never_puts_a_board_or_session_on_both_sides():
    rng = np.random.default_rng(11)
    records = [_rec(f"f{i}", f"B{rng.integers(30)}", f"S{rng.integers(60)}") for i in range(300)]
    for frac in (0.1, 0.3, 0.5, 0.9):
        group_of, members, _ = ve.group_records(records)
        side = ve.split_groups(members, frac, "seed")
        assert ve.check_split_purity(records, group_of, side) == []
        assert set(side.values()) <= {"test", "train_reserved"}


def test_purity_check_catches_a_frame_level_split():
    """The guard itself: split two frames of one board by frame, as a naive
    random split would, and it must object."""
    records = [_rec("a", "B1"), _rec("b", "B1")]
    leaky_group_of = {"a": "ga", "b": "gb"}
    problems = ve.check_split_purity(records, leaky_group_of, {"ga": "test", "gb": "train_reserved"})
    assert problems == ["board:B1 appears in both the held-out and reserved splits"]


def test_held_out_groups_stay_held_out_as_data_grows():
    base = [_rec(f"f{i}", f"B{i}") for i in range(40)]
    _, members, _ = ve.group_records(base)
    before = ve.split_groups(members, 0.3, "seed")
    grown = base + [_rec(f"n{i}", f"B{100 + i}", f"S{i}") for i in range(40)]
    grown += [_rec("link0", "B0", "Slink"), _rec("link1", "B1", "Slink")]  # merges two old groups
    group_of, members2, _ = ve.group_records(grown)
    after = ve.split_groups(members2, 0.3, "seed")
    for rec in base:
        if before[ve.group_records([rec])[0][rec["_key"]]] == "test":
            assert after[group_of[rec["_key"]]] == "test"


def test_frames_without_board_or_session_are_excluded_and_counted(tmp_path):
    _add(tmp_path, "align", _rect_image(9), {"delta_theta_deg": 1.0}, board="B1")
    _add(tmp_path, "align", _rect_image(10), {"delta_theta_deg": 1.0})
    report = ve.evaluate(tmp_path, aligners={"x": lambda img: {"available": True, "delta_theta_deg": 1.0}},
                         cfg=_cfg())
    assert report["split"]["n_excluded_no_board_or_session_id"] == 1
    assert report["alignment"]["n_frames"] == 1


def test_refuses_when_a_held_out_board_was_trained_on(tmp_path):
    _add(tmp_path, "align", _rect_image(11), {"delta_theta_deg": 1.0}, board="B7", session="S1")
    with pytest.raises(ve.EvalRefused, match="training data"):
        ve.evaluate(tmp_path, cfg=_cfg(), trained_on={"board_ids": ["B7"]})
    with pytest.raises(ve.EvalRefused, match="training data"):
        ve.evaluate(tmp_path, cfg=_cfg(), trained_on={"session_ids": ["S1"]})


def test_refuses_one_image_filed_under_two_boards(tmp_path):
    img = _rect_image(12)
    _add(tmp_path, "align", img, {"delta_theta_deg": 1.0}, board="B1")
    _add(tmp_path, "align", img, {"delta_theta_deg": 1.0}, board="B2")
    with pytest.raises(ve.EvalRefused, match="identical frames"):
        ve.evaluate(tmp_path, cfg=_cfg())


def test_near_duplicates_across_the_split_are_reported(tmp_path):
    cfg = _cfg(test_fraction=0.5)
    img = _rect_image(13)
    arr = np.array(img)
    arr[0, 0] = (0, 0, 0)  # one pixel off: not identical, still a near duplicate
    _add(tmp_path, "align", img, {"delta_theta_deg": 1.0}, board=_board_on_side("test", cfg))
    _add(tmp_path, "align", Image.fromarray(arr), {"delta_theta_deg": 1.0}, board=_board_on_side("train", cfg))
    report = ve.evaluate(tmp_path, cfg=cfg)
    assert report["split"]["cross_split_near_duplicates"]["count"] == 1


def test_renders_and_dataset_samples_are_never_evaluated(tmp_path):
    """The frontend tags each frame's source. A 3D-scene render or a bundled
    public-dataset sample isn't a photo of a SERC board, even if someone
    labels it and gives it a board id."""
    def add(seed, source, board):
        cid = dc.save_capture("align", _rect_image(seed), {}, base_dir=str(tmp_path), board_id=board,
                              context={**SYNTHETIC, "frame_source": source})
        dc.record_feedback("align", cid, {"delta_theta_deg": 0.0}, base_dir=str(tmp_path),
                           label_source="human", labeler="fixture-labeler")
    add(800, "camera", "B1")
    add(801, "scene_render", "B2")
    add(802, "sample", "B3")
    report = ve.evaluate(tmp_path, cfg=_cfg(), aligners={"x": lambda img: {"available": True, "delta_theta_deg": 0.0}})
    assert report["alignment"]["n_frames"] == 1
    assert report["split"]["n_excluded_not_serc_photo"] == 2
    assert report["split"]["labeled_frames_by_source"] == {"camera": 1, "scene_render": 1, "sample": 1}


def test_split_manifest_partitions_ids(tmp_path):
    cfg = _cfg(test_fraction=0.5)
    test_board, train_board = _board_on_side("test", cfg), _board_on_side("train", cfg)
    _add(tmp_path, "align", _rect_image(14), {"delta_theta_deg": 0.0}, board=test_board)
    _add(tmp_path, "detect", _rect_image(15), {"boxes": []}, board=train_board)
    m = ve.split_manifest(tmp_path, cfg)
    assert m["test"]["board_ids"] == [test_board]
    assert m["train_reserved"]["board_ids"] == [train_board]
    assert not set(m["test"]["capture_ids"]) & set(m["train_reserved"]["capture_ids"])


# ─────────────────────────────────────────────────────────────────────────
# End to end on synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────

def test_detection_report_matches_hand_counts(tmp_path):
    preds = {}
    seed = 100
    for b in range(6):  # 6 boards, so CIs resample boards
        for f in range(2):
            img = _rect_image(seed)
            seed += 1
            gt = [{"class": "R", "box": [55, 48, 105, 72]}, {"class": "C", "box": [5, 5, 25, 25]}]
            p = [{"class": "R", "box": [56, 48, 105, 72], "score": 0.9}]             # R found
            if b < 3:
                p.append({"class": "C", "box": [5, 5, 25, 25], "score": 0.8})        # C found on half the boards
            if f == 0:
                p.append({"class": "LED", "box": [120, 80, 140, 100], "score": 0.3})  # one false LED per board
            preds[ve._pixel_digest(img)] = p
            _add(tmp_path, "detect", img, {"boxes": gt}, board=f"B{b}", session=f"S{b}")

    report = ve.evaluate(tmp_path, detectors={"fake": _lookup(preds)}, cfg=_cfg(), known_classes=["R", "C"])
    assert report["data_provenance"] == "SYNTHETIC_TEST_FIXTURE"
    d = report["detection"]["detectors"]["fake"]
    assert d["n_frames"] == 12 and d["n_groups"] == 6
    pooled = d["pooled_all_classes"]
    # tp: 12 R + 6 C = 18; fp: 6 LED; fn: 6 C
    assert (pooled["tp"], pooled["fp"], pooled["fn"]) == (18, 6, 6)
    assert pooled["precision"]["value"] == 0.75 and pooled["recall"]["value"] == 0.75
    assert d["per_class"]["R"]["recall"]["value"] == 1.0
    assert d["per_class"]["C"]["recall"]["value"] == 0.5
    assert d["per_class"]["LED"]["precision"]["value"] == 0.0
    assert d["per_class"]["LED"]["recall"]["value"] is None
    assert "labeled_classes_outside_model_taxonomy" not in d
    c_recall = d["per_class"]["C"]["recall"]
    lo, hi = c_recall["ci95_wilson_per_box"]
    assert lo < 0.5 < hi
    blo, bhi = c_recall["ci95_bootstrap"]
    assert blo <= 0.5 <= bhi and c_recall["bootstrap_method"] == "board_bootstrap"
    # boards either found C on both frames or neither, so resampling boards
    # gives a wider interval than treating the 12 boxes as independent
    assert (bhi - blo) > (hi - lo)


def test_detection_score_threshold_drops_low_scores(tmp_path):
    img = _rect_image(200)
    preds = {ve._pixel_digest(img): [{"class": "R", "box": [55, 48, 105, 72], "score": 0.2}]}
    _add(tmp_path, "detect", img, {"boxes": [{"class": "R", "box": [55, 48, 105, 72]}]}, board="B1")
    report = ve.evaluate(tmp_path, detectors={"fake": _lookup(preds)}, cfg=_cfg(score_threshold=0.5))
    pooled = report["detection"]["detectors"]["fake"]["pooled_all_classes"]
    assert (pooled["tp"], pooled["fn"], pooled["precision"]["value"]) == (0, 1, None)


def test_alignment_report_compares_two_paths_on_the_same_frames(tmp_path):
    truth, seed = {}, 300
    for b in range(5):
        for f in range(3):
            img = _rect_image(seed)
            seed += 1
            gt = {"delta_theta_deg": float(b * 3 - f), "delta_x_mm": 0.1 * f, "delta_y_mm": -0.05 * b}
            truth[ve._pixel_digest(img)] = gt
            _add(tmp_path, "align", img, gt, board=f"B{b}", px_per_mm=20.0)

    def exact(img):
        gt = truth[ve._pixel_digest(img)]
        return {"available": True, "delta_theta_deg": gt["delta_theta_deg"],
                "delta_x_px": gt["delta_x_mm"] * 20.0, "delta_y_px": gt["delta_y_mm"] * 20.0}

    def biased(img):
        gt = truth[ve._pixel_digest(img)]
        if gt["delta_theta_deg"] == 0.0:
            return {"available": False}
        return {"available": True, "delta_theta_deg": gt["delta_theta_deg"] + 2.0,
                "delta_x_px": (gt["delta_x_mm"] + 0.5) * 20.0, "delta_y_px": gt["delta_y_mm"] * 20.0}

    report = ve.evaluate(tmp_path, aligners={"exact": exact, "biased": biased}, cfg=_cfg())
    al = report["alignment"]
    assert al["px_per_mm_sources"] == {"capture_context": 15}
    ex, bi = al["methods"]["exact"], al["methods"]["biased"]
    assert ex["angle_abs_error_deg"]["mean"]["value"] == 0.0
    assert ex["position_error_mm"]["mean"]["value"] == 0.0
    assert bi["coverage"]["answered"] == 14 and bi["coverage"]["of"] == 15
    assert bi["angle_abs_error_deg"]["mean"]["value"] == 2.0
    assert bi["angle_abs_error_deg"]["mean"]["ci95"] == [2.0, 2.0]
    assert bi["position_error_mm"]["p95"]["value"] == 0.5
    assert bi["position_bias_mm"]["x"]["value"] == 0.5
    diff = al["paired_difference"]
    assert diff["b_minus_a"] == "biased - exact" and diff["n_frames_both_answered"] == 14
    assert diff["angle_abs_error_deg_mean_diff"]["value"] == 2.0
    assert diff["position_error_mm_mean_diff"]["value"] == 0.5


def test_alignment_without_any_scale_reports_angle_only(tmp_path):
    _add(tmp_path, "align", _rect_image(400), {"delta_theta_deg": 5.0, "delta_x_px": 3.0, "delta_y_px": 0.0},
         board="B1")
    report = ve.evaluate(tmp_path, cfg=_cfg(), aligners={
        "x": lambda img: {"available": True, "delta_theta_deg": 4.0, "delta_x_px": 3.0, "delta_y_px": 0.0}})
    m = report["alignment"]["methods"]["x"]
    assert m["angle_abs_error_deg"]["mean"]["value"] == 1.0
    assert m["position_error_mm"]["n"] == 0 and "not measured" in m["position_error_mm"]["note"]
    assert report["alignment"]["px_per_mm_sources"] == {"none": 1}
    assert "ci_warning" in m  # one board: interval resamples frames and says so


def test_px_per_mm_needs_a_stated_source(tmp_path):
    with pytest.raises(ValueError, match="px_per_mm_source"):
        ve.evaluate(tmp_path, cfg=_cfg(px_per_mm=10.0))


def test_real_classical_path_on_synthetic_rectangles(tmp_path):
    """The production classical-CV aligner on drawn rectangles of known
    pose: a sanity check of the harness wiring, not a SERC result."""
    import vision_classical
    px_per_mm, seed = 10.0, 500
    for b in range(5):
        for f in range(2):
            angle, dx, dy = -20.0 + 8 * b + 3 * f, 6.0 * f - 3, 4.0 - 2 * b
            img = _rect_image(seed, center=(80 + dx, 60 + dy), angle=angle)
            seed += 1
            _add(tmp_path, "align", img, {"delta_theta_deg": angle, "delta_x_px": dx, "delta_y_px": dy},
                 board=f"B{b}", px_per_mm=px_per_mm)
    report = ve.evaluate(tmp_path, cfg=_cfg(), aligners={"classical_cv": vision_classical.estimate_alignment})
    m = report["alignment"]["methods"]["classical_cv"]
    assert report["data_provenance"] == "SYNTHETIC_TEST_FIXTURE"
    assert m["coverage"]["answered"] == 10
    assert m["angle_abs_error_deg"]["mean"]["value"] < 1.5
    assert m["position_error_mm"]["mean"]["value"] < 0.2


def test_empty_capture_dir_reports_no_measurement(tmp_path):
    report = ve.evaluate(tmp_path, detectors={"d": lambda img: []}, aligners={"a": lambda img: {}}, cfg=_cfg())
    assert report["detection"]["detectors"]["d"] == {"status": "no_labeled_test_frames"}
    assert report["alignment"] == {"status": "no_labeled_test_frames"}
    assert report["split"]["n_test_frames_evaluated"] == 0


# ─────────────────────────────────────────────────────────────────────────
# CLI with the production models
# ─────────────────────────────────────────────────────────────────────────

def test_cli_writes_a_synthetic_labeled_report(tmp_path, capsys):
    caps = tmp_path / "caps"
    _add(caps, "align", _rect_image(600, angle=10), {"delta_theta_deg": 10.0, "delta_x_px": 0.0, "delta_y_px": 0.0},
         board="B1", px_per_mm=10.0)
    _add(caps, "detect", _rect_image(601), {"boxes": [{"class": "R", "box": [55, 48, 105, 72]}]}, board="B2")
    out, split = tmp_path / "report.json", tmp_path / "split.json"
    code = ve.main(["--capture-dir", str(caps), "--out", str(out), "--test-fraction", "1.0",
                    "--detectors", "yolo_hybrid,sliding_window", "--n-boot", "100", "--write-split", str(split)])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["data_provenance"] == "SYNTHETIC_TEST_FIXTURE"
    dets = report["detection"]["detectors"]
    assert dets["sliding_window"]["status"] == "measured"
    assert dets["yolo_hybrid"]["status"] in ("measured", "not_run")  # not_run where ultralytics isn't installed
    if dets["yolo_hybrid"]["status"] == "not_run":
        assert dets["yolo_hybrid"]["reason"]
    assert set(report["alignment"]["methods"]) == {"classical_cv", "jepa_synthetic_trained"}
    assert report["models"]["classifier"]["exists"] and len(report["models"]["classifier"]["sha256"]) == 64
    assert json.loads(split.read_text())["test"]["board_ids"] == ["B1", "B2"]
    assert "SYNTHETIC_TEST_FIXTURE" in capsys.readouterr().out


def test_cli_refuses_and_writes_nothing_for_model_labels(tmp_path, capsys):
    caps = tmp_path / "caps"
    _add(caps, "align", _rect_image(700), {"delta_theta_deg": 1.0}, board="B1", label_source="model")
    out = tmp_path / "report.json"
    code = ve.main(["--capture-dir", str(caps), "--out", str(out), "--test-fraction", "1.0", "--no-detection"])
    assert code == 2
    assert not out.exists()
    assert "REFUSED" in capsys.readouterr().err
