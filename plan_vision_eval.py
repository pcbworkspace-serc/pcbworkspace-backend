"""
plan_vision_eval.py — how many boards to label before vision_eval.py's
intervals are useful. A planning aid, not a measurement: it simulates
labeled data under the assumptions below and runs the harness's own
board-bootstrap interval code on it.

Assumptions (change them to match what you expect):
  detection: true recall 0.85 that varies from board to board (sd 0.6 on
             the logit scale), 20 labeled parts per board
  alignment: per-frame angle error sd 2.0 deg plus a per-board offset with
             sd 1.0 deg, 10 labeled frames per board

Usage:
    python plan_vision_eval.py
"""

import numpy as np

import vision_eval as ve

REPS = 300
BOARDS = (5, 10, 20, 40)


def recall_halfwidth(rng, cfg, boards, parts_per_board, recall=0.85, board_sd=0.6):
    """Median half-width of the board-bootstrap and Wilson intervals."""
    boot, wilson = [], []
    logit = np.log(recall / (1 - recall))
    groups = [g for g in range(boards) for _ in range(parts_per_board)]
    for _ in range(REPS):
        per_board = 1 / (1 + np.exp(-(logit + rng.normal(0, board_sd, boards))))
        hits = np.concatenate([rng.random(parts_per_board) < p for p in per_board]).astype(float)
        res = ve.Resampler(groups, cfg, rng)
        lo, hi = res.ci((res.weights @ hits) / res.weights.sum(axis=1))
        boot.append((hi - lo) / 2)
        lo, hi = ve.wilson_interval(int(hits.sum()), len(hits))
        wilson.append((hi - lo) / 2)
    return float(np.median(boot)), float(np.median(wilson))


def angle_halfwidth(rng, cfg, boards, frames_per_board, frame_sd=2.0, board_sd=1.0):
    """Median half-width of the board-bootstrap interval on mean |angle error|."""
    widths = []
    groups = [g for g in range(boards) for _ in range(frames_per_board)]
    for _ in range(REPS):
        errs = np.concatenate([
            np.abs(rng.normal(rng.normal(0, board_sd), frame_sd, frames_per_board)) for _ in range(boards)
        ])
        res = ve.Resampler(groups, cfg, rng)
        lo, hi = res.ci((res.weights @ errs) / res.weights.sum(axis=1))
        widths.append((hi - lo) / 2)
    return float(np.median(widths))


def main():
    rng = np.random.default_rng(42)
    cfg = ve.EvalConfig(n_boot=600)
    print("Planning simulation (assumptions in this file's docstring; not a measurement)")
    print(f"{'held-out boards':>16} | {'recall interval':>16} | {'Wilson, ignoring boards':>23} | {'mean angle-error interval':>25}")
    for boards in BOARDS:
        boot, wilson = recall_halfwidth(rng, cfg, boards, 20)
        angle = angle_halfwidth(rng, cfg, boards, 10)
        print(f"{boards:>16} | {'+/-%.1f pts' % (boot * 100):>16} | {'+/-%.1f pts' % (wilson * 100):>23} | {'+/-%.2f deg' % angle:>25}")


if __name__ == "__main__":
    main()
