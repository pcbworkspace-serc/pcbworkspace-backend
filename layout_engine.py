"""
layout_engine.py — pure-Python circuit/robot-safety logic for PCBWorkspace.

No Flask, no network, no ML weights — everything here is a plain function
over plain dicts so it can be unit tested without a server or an API key.

Four pieces:
  clamp_vla_action   — reject/clamp a single robot action before it reaches
                        real motors (server-side; do not trust the LLM prompt)
  run_drc_erc        — connectivity checks over a structured netlist
                        (floating nets, duplicate refs, pin-count mismatches)
  auto_place         — simulated-annealing component placement (HPWL cost)
  auto_route         — grid-based Lee/BFS maze router, one net at a time
"""

import math
import random

ALLOWED_ACTIONS = {"home", "move", "rotate", "pick", "place", "release",
                    "scan", "detect", "align", "validate"}


# ─────────────────────────────────────────────────────────────────────────
# VLA action safety clamping
# ─────────────────────────────────────────────────────────────────────────

def clamp_vla_action(action, board_w, board_h, z_max=20.0, margin=5.0):
    """Validate + clamp one action dict from the LLM before it is ever sent
    to a motor. Returns (clamped_action_or_None, warning_or_None).

    Values within `margin` mm of the board are clamped in (small overshoot,
    likely a rounding/edge case). Values further out are dropped entirely —
    a large out-of-range coordinate is far more likely a hallucination than
    an intentional off-board move, and this arm should never chase one.
    """
    if not isinstance(action, dict):
        return None, "dropped non-dict action"

    kind = action.get("action")
    if kind not in ALLOWED_ACTIONS:
        return None, "dropped unknown action type: %r" % (kind,)

    out = {"action": kind}

    if kind == "move":
        try:
            x = float(action["x_mm"]); y = float(action["y_mm"]); z = float(action["z_mm"])
        except (KeyError, TypeError, ValueError):
            return None, "dropped move - missing/non-numeric x_mm/y_mm/z_mm"

        if not (-margin <= x <= board_w + margin):
            return None, "dropped move - x_mm=%.2f outside safety envelope [-%.1f, %.1f]" % (x, margin, board_w + margin)
        if not (-margin <= y <= board_h + margin):
            return None, "dropped move - y_mm=%.2f outside safety envelope [-%.1f, %.1f]" % (y, margin, board_h + margin)
        if not (-margin <= z <= z_max + margin):
            return None, "dropped move - z_mm=%.2f outside safety envelope [-%.1f, %.1f]" % (z, margin, z_max + margin)

        cx = min(max(x, 0.0), board_w)
        cy = min(max(y, 0.0), board_h)
        cz = min(max(z, 0.0), z_max)
        out["x_mm"] = round(cx, 3); out["y_mm"] = round(cy, 3); out["z_mm"] = round(cz, 3)
        warn = None
        if (cx, cy, cz) != (x, y, z):
            warn = "clamped move (%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f)" % (x, y, z, cx, cy, cz)
        return out, warn

    if kind == "rotate":
        try:
            deg = float(action["degrees"])
        except (KeyError, TypeError, ValueError):
            return None, "dropped rotate - missing/non-numeric degrees"
        cdeg = max(-180.0, min(180.0, deg))
        out["degrees"] = round(cdeg, 2)
        return out, ("clamped rotate %.2f -> %.2f" % (deg, cdeg)) if cdeg != deg else None

    # home | pick | place | release | scan | detect | align | validate — no fields to check
    return out, None


def clamp_vla_plan(actions, board_w, board_h, z_max=20.0, margin=5.0):
    """Clamp a whole action list. Returns (clean_actions, warnings)."""
    clean, warnings = [], []
    for a in (actions or []):
        c, w = clamp_vla_action(a, board_w, board_h, z_max=z_max, margin=margin)
        if w:
            warnings.append(w)
        if c is not None:
            clean.append(c)
    return clean, warnings


# ─────────────────────────────────────────────────────────────────────────
# DRC / ERC — structured netlist connectivity checks
# ─────────────────────────────────────────────────────────────────────────

EXPECTED_PIN_COUNTS = {
    "resistor": 2, "capacitor": 2, "inductor": 2, "diode": 2, "led": 2,
    "crystal": 2, "fuse": 2, "switch": 2,
    "transistor": 3, "mosfet": 3, "regulator": 3,
    "connector": None,  # variable pin count — skip check
    "ic": None,
}

GROUND_NAMES = {"gnd", "ground", "0v", "agnd", "dgnd", "vss"}


def run_drc_erc(circuit):
    """circuit = {"components": [{"ref","type","value","footprint","pins": {pin_no: net_name}}]}
    Returns {"errors": [...], "warnings": [...], "clean": bool, "net_count", "component_count"}.

    This exists to catch LLM connectivity hallucination mechanically instead
    of trusting the model's own claim that a generated schematic is correct.
    """
    errors, warnings = [], []
    components = circuit.get("components") or []

    seen_refs = set()
    net_members = {}  # net_name -> [(ref, pin_no), ...]

    for comp in components:
        ref = comp.get("ref")
        ctype = (comp.get("type") or "").strip().lower()
        pins = comp.get("pins") or {}

        if not ref:
            errors.append("component missing 'ref'")
            continue
        if ref in seen_refs:
            errors.append("duplicate reference designator: %s" % ref)
        seen_refs.add(ref)

        if not pins:
            errors.append("%s has no pins defined" % ref)
            continue

        expected = EXPECTED_PIN_COUNTS.get(ctype)
        if expected is not None and len(pins) != expected:
            errors.append("%s (%s) has %d pins, expected %d" % (ref, ctype, len(pins), expected))

        pin_nets = []
        for pin_no, net in pins.items():
            if net is None or str(net).strip() == "":
                errors.append("%s pin %s is not connected to any net" % (ref, pin_no))
                continue
            net = str(net).strip()
            pin_nets.append(net)
            net_members.setdefault(net, []).append((ref, pin_no))

        # both pins of a 2-pin passive tied to the same net = dead component
        if ctype in ("resistor", "capacitor", "inductor", "diode", "led", "fuse") \
                and len(pin_nets) >= 2 and len(set(pin_nets)) == 1:
            warnings.append("%s - both pins on net '%s', component has no effect" % (ref, pin_nets[0]))

    for net, members in net_members.items():
        if len(members) < 2:
            errors.append("floating net '%s' - only 1 connection (%s.%s)" % (net, members[0][0], members[0][1]))

    if components and not any(n.lower() in GROUND_NAMES for n in net_members):
        warnings.append("no ground net found (expected one of: %s)" % ", ".join(sorted(GROUND_NAMES)))

    return {
        "errors": errors,
        "warnings": warnings,
        "clean": len(errors) == 0,
        "net_count": len(net_members),
        "component_count": len(components),
    }


# ─────────────────────────────────────────────────────────────────────────
# Auto-placement — simulated annealing over HPWL + overlap cost
# ─────────────────────────────────────────────────────────────────────────

FOOTPRINT_SIZE_MM = {
    "0402": (1.0, 0.5), "0603": (1.6, 0.8), "0805": (2.0, 1.25),
    "sot-23": (3.0, 1.5), "sot-223": (6.5, 3.5),
    "soic-8": (5.0, 4.0), "qfp-32": (9.0, 9.0), "dip-8": (9.5, 7.6),
}
DEFAULT_SIZE_MM = (4.0, 4.0)


def _footprint_size(comp):
    fp = (comp.get("footprint") or "").strip().lower()
    return FOOTPRINT_SIZE_MM.get(fp, DEFAULT_SIZE_MM)


def _nets_from_components(components):
    nets = {}
    for comp in components:
        ref = comp.get("ref")
        for pin_no, net in (comp.get("pins") or {}).items():
            if net:
                nets.setdefault(str(net), set()).add(ref)
    return {n: sorted(refs) for n, refs in nets.items() if len(refs) >= 2}


def _hpwl_cost(positions, nets):
    cost = 0.0
    for refs in nets.values():
        xs = [positions[r][0] for r in refs if r in positions]
        ys = [positions[r][1] for r in refs if r in positions]
        if xs and ys:
            cost += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return cost


def _overlap_penalty(positions, sizes, pad_mm=0.5):
    refs = list(positions.keys())
    penalty = 0.0
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            r1, r2 = refs[i], refs[j]
            x1, y1 = positions[r1]; x2, y2 = positions[r2]
            w1, h1 = sizes[r1]; w2, h2 = sizes[r2]
            dx = abs(x1 - x2) - (w1 + w2) / 2 - pad_mm
            dy = abs(y1 - y2) - (h1 + h2) / 2 - pad_mm
            if dx < 0 and dy < 0:
                penalty += min(-dx, -dy) * 20.0  # overlap is much worse than long wires
    return penalty


def auto_place(components, board_w, board_h, iterations=3000, seed=None, margin=2.0):
    """Simulated-annealing placement. Returns
    {"positions": {ref: [x,y]}, "cost": float, "iterations": int}.
    """
    rng = random.Random(seed)
    refs = [c["ref"] for c in components if c.get("ref")]
    if not refs:
        return {"positions": {}, "cost": 0.0, "iterations": 0}

    sizes = {c["ref"]: _footprint_size(c) for c in components if c.get("ref")}
    nets = _nets_from_components(components)

    positions = {}
    for ref in refs:
        w, h = sizes[ref]
        positions[ref] = [
            rng.uniform(margin + w / 2, max(margin + w / 2, board_w - margin - w / 2)),
            rng.uniform(margin + h / 2, max(margin + h / 2, board_h - margin - h / 2)),
        ]

    def cost_of(pos):
        return _hpwl_cost(pos, nets) + _overlap_penalty(pos, sizes)

    current_cost = cost_of(positions)
    T0, T1 = 8.0, 0.05
    for it in range(iterations):
        T = T0 * ((T1 / T0) ** (it / max(1, iterations - 1)))
        ref = rng.choice(refs)
        w, h = sizes[ref]
        old = positions[ref][:]
        step = max(0.5, T)
        positions[ref][0] = min(max(positions[ref][0] + rng.uniform(-step, step), margin + w / 2), max(margin + w / 2, board_w - margin - w / 2))
        positions[ref][1] = min(max(positions[ref][1] + rng.uniform(-step, step), margin + h / 2), max(margin + h / 2, board_h - margin - h / 2))

        new_cost = cost_of(positions)
        if new_cost <= current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, 1e-6)):
            current_cost = new_cost
        else:
            positions[ref] = old

    return {
        "positions": {r: [round(p[0], 3), round(p[1], 3)] for r, p in positions.items()},
        "sizes_mm": {r: list(sizes[r]) for r in refs},
        "cost": round(current_cost, 3),
        "iterations": iterations,
    }


# ─────────────────────────────────────────────────────────────────────────
# Auto-routing — grid-based Lee/BFS maze router
# ─────────────────────────────────────────────────────────────────────────

def _mst_edges(refs, positions):
    """Minimum spanning tree over pin positions (Prim's) so a >2-pin net
    routes as a tree of point-to-point segments instead of a full mesh."""
    if len(refs) < 2:
        return []
    in_tree = {refs[0]}
    edges = []
    remaining = set(refs[1:])
    while remaining:
        best = None
        for a in in_tree:
            for b in remaining:
                d = math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        edges.append((a, b))
        in_tree.add(b)
        remaining.discard(b)
    return edges


def _bfs_route(grid_w, grid_h, blocked, start, goal):
    """4-connected BFS maze route on a boolean-obstacle grid. Returns a list
    of (gx,gy) cells, or None if unreachable."""
    from collections import deque
    if start == goal:
        return [start]
    seen = {start}
    prev = {}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                continue
            if (nx, ny) in seen or ((nx, ny) in blocked and (nx, ny) != goal):
                continue
            seen.add((nx, ny))
            prev[(nx, ny)] = (cx, cy)
            if (nx, ny) == goal:
                path = [goal]
                node = goal
                while node != start:
                    node = prev[node]
                    path.append(node)
                path.reverse()
                return path
            q.append((nx, ny))
    return None


def auto_route(positions, components, board_w, board_h, grid_mm=1.0, keepout_mm=1.0):
    """Route every multi-pin net as an MST of point-to-point BFS paths on a
    Manhattan grid. Obstacles are component keepout footprints only — this
    is a single-pass v1 router: it does NOT prevent one net's trace from
    crossing another's (that needs a real via/layer model), it just flags
    crossings so a human can add a via or move to layer 2.
    """
    grid_w = max(1, int(math.ceil(board_w / grid_mm)))
    grid_h = max(1, int(math.ceil(board_h / grid_mm)))

    def to_cell(x, y):
        return (int(round(x / grid_mm)), int(round(y / grid_mm)))

    sizes = {c["ref"]: _footprint_size(c) for c in components if c.get("ref")}
    keepout_by_ref = {}
    pin_cells = {}
    for ref, (x, y) in positions.items():
        pin_cells[ref] = to_cell(x, y)
        w, h = sizes.get(ref, DEFAULT_SIZE_MM)
        kx = int(math.ceil((w / 2 + keepout_mm) / grid_mm))
        ky = int(math.ceil((h / 2 + keepout_mm) / grid_mm))
        cx, cy = pin_cells[ref]
        keepout_by_ref[ref] = {(gx, gy) for gx in range(cx - kx, cx + kx + 1)
                                         for gy in range(cy - ky, cy + ky + 1)}
    blocked = set().union(*keepout_by_ref.values()) if keepout_by_ref else set()

    nets = _nets_from_components(components)
    routed, unrouted = [], []
    all_used_cells = {}  # cell -> net name, for crossing detection

    for net, refs in nets.items():
        edges = _mst_edges(refs, positions)
        net_paths = []
        ok = True
        for a, b in edges:
            start, goal = pin_cells[a], pin_cells[b]
            # a wire has to be able to leave its own component's keepout, and
            # arrive inside the target's — only those two are excluded here,
            # every other component (including others on this same net) still
            # blocks, so the route has to go around them
            local_blocked = blocked - keepout_by_ref.get(a, set()) - keepout_by_ref.get(b, set())
            path = _bfs_route(grid_w, grid_h, local_blocked, start, goal)
            if path is None:
                ok = False
                continue
            net_paths.append({"from": a, "to": b, "cells": path,
                               "mm": [[round(gx * grid_mm, 2), round(gy * grid_mm, 2)] for gx, gy in path]})
            for cell in path:
                if cell in all_used_cells and all_used_cells[cell] != net:
                    pass  # crossing — surfaced below via crossings list
                all_used_cells.setdefault(cell, net)
        if ok and net_paths:
            routed.append({"net": net, "segments": net_paths})
        else:
            unrouted.append(net)

    # crossing detection: any cell touched by >1 distinct net
    cell_nets = {}
    for entry in routed:
        for seg in entry["segments"]:
            for cell in seg["cells"]:
                cell_nets.setdefault(cell, set()).add(entry["net"])
    crossings = [{"cell_mm": [round(c[0] * grid_mm, 2), round(c[1] * grid_mm, 2)], "nets": sorted(n)}
                 for c, n in cell_nets.items() if len(n) > 1]

    return {
        "routed_nets": routed,
        "unrouted_nets": unrouted,
        "crossings_needing_via": crossings,
        "grid_mm": grid_mm,
        "layer_note": "v1 heuristic single-pass router - crossings need a via/layer-2 jump, not yet DRC-clearance verified for fab",
    }
