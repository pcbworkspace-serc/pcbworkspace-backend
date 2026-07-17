"""
layout_engine.py — pure-Python circuit/robot-safety logic for PCBWorkspace.

No Flask, no network, no ML weights — everything here is a plain function
over plain dicts so it can be unit tested without a server or an API key.

Five pieces:
  clamp_vla_action    — reject/clamp a single robot action before it reaches
                        real motors (server-side; do not trust the LLM prompt)
  filter_unreachable  — a second, geometric safety net: a target can be
                        inside the board rectangle and still be outside
                        what the arm's own kinematics can reach
  run_drc_erc         — connectivity checks over a structured netlist
                        (floating nets, duplicate refs, pin-count mismatches)
  auto_place          — simulated-annealing component placement (HPWL cost)
  auto_route          — pin-to-pin Lee/BFS maze router across 1-2 layers,
                        resolving net crossings with a layer hop where
                        possible instead of only flagging them
"""

import math
import random

ALLOWED_ACTIONS = {"home", "move", "rotate", "pick", "place", "release",
                    "scan", "detect", "align", "validate"}

# 2-link planar arm geometry, straight from the Rev 2 hardware doc's arm
# segment lengths (150mm base->elbow, 180mm elbow->nozzle) — same numbers
# the reach-simulation artifact uses, so the two stay consistent.
ARM_L1_MM = 150.0
ARM_L2_MM = 180.0

# Where the board's own (0,0) sits in the arm's base frame. THIS IS A
# PLACEHOLDER, not a measurement — nothing in this codebase has calibrated
# the physical offset between the board origin and the arm base yet. It's
# set to the same value the reach-sim artifact uses so the two don't
# silently disagree, but it needs a real number once that calibration
# exists. Until then this check catches only the clearly-impossible cases
# (way outside the reach envelope), which is still worth having — it's a
# safety net layered on top of clamp_vla_action's board-bounds check, not
# a replacement for real arm-frame calibration.
ARM_BASE_OFFSET_MM = (190.0, 40.0)


def check_reach(board_x, board_y, offset=ARM_BASE_OFFSET_MM, l1=ARM_L1_MM, l2=ARM_L2_MM):
    """Is (board_x, board_y) inside the 2-link arm's annular reach envelope?
    Returns (reachable: bool, r: float distance from arm base)."""
    arm_x = board_x + offset[0]
    arm_y = board_y + offset[1]
    r = math.hypot(arm_x, arm_y)
    min_r, max_r = abs(l1 - l2), l1 + l2
    return (min_r <= r <= max_r), r


def filter_unreachable_actions(actions, offset=ARM_BASE_OFFSET_MM, l1=ARM_L1_MM, l2=ARM_L2_MM):
    """Second pass after clamp_vla_plan: drop 'move' actions whose (x_mm,
    y_mm) — while inside the board rectangle — fall outside the arm's own
    kinematic reach. Returns (filtered_actions, warnings)."""
    kept, warnings = [], []
    for action in actions:
        if action.get("action") != "move":
            kept.append(action)
            continue
        reachable, r = check_reach(action["x_mm"], action["y_mm"], offset=offset, l1=l1, l2=l2)
        if reachable:
            kept.append(action)
        else:
            min_r, max_r = abs(l1 - l2), l1 + l2
            warnings.append(
                "dropped move (%.1f,%.1f) - r=%.1fmm outside arm reach envelope [%.1f,%.1f]mm"
                " (ARM_BASE_OFFSET_MM is an unmeasured placeholder - verify against real calibration)"
                % (action["x_mm"], action["y_mm"], r, min_r, max_r)
            )
    return kept, warnings


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
# BOM plausibility — a substitute for live Octopart/Digikey sourcing
# ─────────────────────────────────────────────────────────────────────────
# There's no sourcing API key available to check that a generated part
# actually exists and is buyable — that needs a real Octopart/Digikey/
# Mouser key this environment doesn't have, and shipping an untested
# integration against a service nothing here can actually call would be
# worse than not having it. This is the deterministic stand-in: catch the
# adjacent problem DRC can't — a value that doesn't look like a real part
# value for its type ("10uF" on a part typed "resistor"), a missing value
# on a part that needs one, or a footprint code nothing in this codebase
# recognizes. It won't catch "this exact part number doesn't exist" —
# only "this value/footprint combination doesn't make physical sense."

import re

_RESISTOR_VALUE_RE = re.compile(r"^\d+(\.\d+)?\s*(m|k|meg|M|R|ohm|ohms|Ω)?$", re.IGNORECASE)
_CAPACITOR_VALUE_RE = re.compile(r"^\d+(\.\d+)?\s*(p|n|u|µ|m)?F$", re.IGNORECASE)
_INDUCTOR_VALUE_RE = re.compile(r"^\d+(\.\d+)?\s*(p|n|u|µ|m)?H$", re.IGNORECASE)

_VALUE_CHECKS = {
    "resistor": (_RESISTOR_VALUE_RE, "a resistance (e.g. '10k', '4.7k', '220', '1M')"),
    "capacitor": (_CAPACITOR_VALUE_RE, "a capacitance (e.g. '100nF', '10uF', '22pF')"),
    "inductor": (_INDUCTOR_VALUE_RE, "an inductance (e.g. '10uH', '1mH')"),
}


def check_bom_plausibility(circuit):
    """Deterministic heuristic checks over value/footprint sanity.
    Returns {"warnings": [...], "plausible": bool}. Never raises on
    malformed input - a missing/odd field is itself something to warn
    about, not a reason to crash the check."""
    warnings = []
    known_footprints = set(FOOTPRINT_SIZE_MM.keys())

    for comp in circuit.get("components") or []:
        ref = comp.get("ref") or "?"
        ctype = (comp.get("type") or "").strip().lower()
        value = (comp.get("value") or "").strip()
        footprint = (comp.get("footprint") or "").strip().lower()

        check = _VALUE_CHECKS.get(ctype)
        if check is not None:
            pattern, description = check
            if not value:
                warnings.append("%s: %s has no value specified" % (ref, ctype))
            elif not pattern.match(value):
                warnings.append("%s: value '%s' doesn't look like %s" % (ref, value, description))

        if footprint and footprint not in known_footprints:
            warnings.append("%s: footprint '%s' isn't one this codebase recognizes - verify it's a real package before ordering" % (ref, footprint))

    return {"warnings": warnings, "plausible": len(warnings) == 0}


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
    """net -> component refs (not pins) - the granularity auto_place's HPWL
    cost needs. auto_route uses the pad-level variant below instead, since
    routing needs to know which pin, not just which component."""
    nets = {}
    for comp in components:
        ref = comp.get("ref")
        for pin_no, net in (comp.get("pins") or {}).items():
            if net:
                nets.setdefault(str(net), set()).add(ref)
    return {n: sorted(refs) for n, refs in nets.items() if len(refs) >= 2}


def _edge_positions(count, span):
    """`count` evenly-spaced offsets along [-span/2, span/2] (0 -> [], 1 -> [0])."""
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    step = span / (count - 1)
    return [-span / 2 + i * step for i in range(count)]


def _pad_offsets(pin_numbers, w, h):
    """(dx, dy) per pin relative to the component's own center — a
    heuristic footprint layout, not a datasheet-exact one: 2-pin parts get
    end-to-end pads, 3-pin parts get a SOT-23-style 2-bottom/1-top layout,
    anything bigger gets a generic dual-row IC layout (left edge top-to-
    bottom, right edge bottom-to-top, matching standard IC pin numbering).
    This only affects routing *geometry* — pin-to-net connectivity (what
    DRC/ERC checks) doesn't depend on where a pad is drawn.
    """
    pins_sorted = sorted(pin_numbers, key=lambda p: (len(str(p)), str(p)))
    n = len(pins_sorted)
    offsets = {}

    if n == 2:
        offsets[pins_sorted[0]] = (-w / 2 * 0.7, 0.0)
        offsets[pins_sorted[1]] = (w / 2 * 0.7, 0.0)
    elif n == 3:
        offsets[pins_sorted[0]] = (-w / 2 * 0.6, -h / 2 * 0.8)
        offsets[pins_sorted[1]] = (w / 2 * 0.6, -h / 2 * 0.8)
        offsets[pins_sorted[2]] = (0.0, h / 2 * 0.8)
    else:
        half = (n + 1) // 2
        left, right = pins_sorted[:half], pins_sorted[half:]
        for p, y in zip(left, _edge_positions(len(left), h * 0.8)):
            offsets[p] = (-w / 2 * 0.85, y)
        for p, y in zip(reversed(right), _edge_positions(len(right), h * 0.8)):
            offsets[p] = (w / 2 * 0.85, y)

    return offsets


def _pad_nets_from_components(components):
    """net -> [(ref, pin_no), ...], pad-level (not component-level)."""
    nets = {}
    for comp in components:
        ref = comp.get("ref")
        for pin_no, net in (comp.get("pins") or {}).items():
            if net:
                nets.setdefault(str(net), []).append((ref, str(pin_no)))
    return {n: members for n, members in nets.items() if len(members) >= 2}


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


def auto_route(positions, components, board_w, board_h, grid_mm=1.0, keepout_mm=1.0, layers=2):
    """Route every multi-pin net as an MST of point-to-point BFS paths on a
    Manhattan grid, pin-to-pin (not centroid-to-centroid — see
    _pad_offsets). Obstacles are component keepouts, always respected.

    Net-to-net crossings are resolved across `layers` routing layers where
    possible: each edge tries layer 1 first, then layer 2, before falling
    back to routing through the conflict anyway (v1's behavior) as a last
    resort so a genuinely congested board still produces a route instead
    of silently failing. Vias are assumed at both ends of any segment that
    lands on layer 2 — this doesn't optimize via placement or do rip-up-
    and-reroute, it's a real improvement over "always flag, never resolve"
    but still not a substitute for a real autorouter before fab.
    """
    grid_w = max(1, int(math.ceil(board_w / grid_mm)))
    grid_h = max(1, int(math.ceil(board_h / grid_mm)))
    layers = max(1, int(layers))

    def to_cell(x, y):
        return (int(round(x / grid_mm)), int(round(y / grid_mm)))

    sizes = {c["ref"]: _footprint_size(c) for c in components if c.get("ref")}
    keepout_by_ref = {}
    for ref, (x, y) in positions.items():
        w, h = sizes.get(ref, DEFAULT_SIZE_MM)
        kx = int(math.ceil((w / 2 + keepout_mm) / grid_mm))
        ky = int(math.ceil((h / 2 + keepout_mm) / grid_mm))
        cx, cy = to_cell(x, y)
        keepout_by_ref[ref] = {(gx, gy) for gx in range(cx - kx, cx + kx + 1)
                                         for gy in range(cy - ky, cy + ky + 1)}
    hard_blocked_all = set().union(*keepout_by_ref.values()) if keepout_by_ref else set()

    # pad-level pin positions: component center + a heuristic per-pin offset
    pad_positions, pad_cells = {}, {}
    for comp in components:
        ref = comp.get("ref")
        if ref not in positions:
            continue
        cx, cy = positions[ref]
        w, h = sizes.get(ref, DEFAULT_SIZE_MM)
        pins = comp.get("pins") or {}
        offsets = _pad_offsets(list(pins.keys()), w, h)
        for pin_no in pins:
            dx, dy = offsets.get(str(pin_no), (0.0, 0.0))
            pad_positions[(ref, str(pin_no))] = (cx + dx, cy + dy)
            pad_cells[(ref, str(pin_no))] = to_cell(cx + dx, cy + dy)

    pad_nets = _pad_nets_from_components(components)
    net_order = sorted(pad_nets.keys(), key=lambda n: (-len(pad_nets[n]), n))  # bigger nets first, then deterministic

    used_by_layer = {layer: {} for layer in range(1, layers + 1)}  # layer -> {cell: net}
    routed, unrouted, crossings = [], [], []

    for net in net_order:
        members = pad_nets[net]
        edges = _mst_edges(members, pad_positions)
        net_paths = []
        net_ok = True

        for a, b in edges:
            ref_a, ref_b = a[0], b[0]
            start, goal = pad_cells[a], pad_cells[b]
            hard_blocked = hard_blocked_all - keepout_by_ref.get(ref_a, set()) - keepout_by_ref.get(ref_b, set())

            path, chosen_layer, via_fallback = None, None, False
            for layer in range(1, layers + 1):
                soft_blocked = {cell for cell, n in used_by_layer[layer].items() if n != net}
                path = _bfs_route(grid_w, grid_h, hard_blocked | soft_blocked, start, goal)
                if path is not None:
                    chosen_layer = layer
                    break

            if path is None:
                # every layer is congested here - route anyway on layer 1,
                # ignoring other nets, and flag the conflict below
                path = _bfs_route(grid_w, grid_h, hard_blocked, start, goal)
                chosen_layer = 1
                via_fallback = True

            if path is None:
                net_ok = False  # geometrically impossible even ignoring other nets
                continue

            for cell in path:
                prior = used_by_layer[chosen_layer].get(cell)
                if prior is not None and prior != net:
                    crossings.append({
                        "cell_mm": [round(cell[0] * grid_mm, 2), round(cell[1] * grid_mm, 2)],
                        "layer": chosen_layer, "nets": sorted([prior, net]),
                    })
                used_by_layer[chosen_layer][cell] = net

            net_paths.append({
                "from": "%s.%s" % a, "to": "%s.%s" % b, "layer": chosen_layer,
                "via_fallback": via_fallback, "cells": path,
                "mm": [[round(gx * grid_mm, 2), round(gy * grid_mm, 2)] for gx, gy in path],
            })

        if net_ok and net_paths:
            routed.append({"net": net, "segments": net_paths})
        else:
            unrouted.append(net)

    return {
        "routed_nets": routed,
        "unrouted_nets": unrouted,
        "crossings_needing_via": crossings,
        "layers_used": layers,
        "grid_mm": grid_mm,
        "layer_note": ("v2: routes pin-to-pin across %d layer(s), resolving net crossings with a layer hop where "
                        "possible. crossings_needing_via now lists only conflicts that couldn't be resolved even "
                        "across all layers, plus via_fallback segments — still not DRC-clearance verified for fab."
                        % layers),
    }
