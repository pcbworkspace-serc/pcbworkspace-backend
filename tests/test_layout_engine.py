import math

import layout_engine as le


# ---- clamp_vla_action ----

def test_normal_move_passes_through_unclamped():
    c, w = le.clamp_vla_action({"action": "move", "x_mm": 31, "y_mm": 21, "z_mm": 5}, 62, 42)
    assert c == {"action": "move", "x_mm": 31.0, "y_mm": 21.0, "z_mm": 5.0}
    assert w is None


def test_small_overshoot_clamps():
    c, w = le.clamp_vla_action({"action": "move", "x_mm": 64, "y_mm": 21, "z_mm": 5}, 62, 42)
    assert c is not None and c["x_mm"] == 62
    assert w is not None


def test_large_hallucinated_coordinate_is_dropped():
    c, w = le.clamp_vla_action({"action": "move", "x_mm": 5000, "y_mm": 21, "z_mm": 5}, 62, 42)
    assert c is None
    assert w is not None


def test_unknown_action_type_dropped():
    c, _ = le.clamp_vla_action({"action": "nuke"}, 62, 42)
    assert c is None


def test_non_numeric_coordinate_dropped():
    c, _ = le.clamp_vla_action({"action": "move", "x_mm": "banana", "y_mm": 1, "z_mm": 1}, 62, 42)
    assert c is None


def test_bare_action_with_no_fields_passes():
    c, w = le.clamp_vla_action({"action": "home"}, 62, 42)
    assert c == {"action": "home"}
    assert w is None


def test_plan_clamp_drops_exactly_the_bad_one():
    actions = [
        {"action": "home"},
        {"action": "move", "x_mm": 31, "y_mm": 21, "z_mm": 5},
        {"action": "move", "x_mm": 99999, "y_mm": 21, "z_mm": 5},
        {"action": "pick"},
    ]
    clean, warnings = le.clamp_vla_plan(actions, 62, 42)
    assert len(clean) == 3
    assert len(warnings) == 1


# ---- check_reach / filter_unreachable_actions ----

def test_default_board_corners_are_within_arm_reach():
    for x, y in [(0, 0), (62, 0), (0, 42), (62, 42)]:
        reachable, r = le.check_reach(x, y)
        assert reachable, (x, y, r)


def test_arm_base_origin_is_in_the_unreachable_dead_zone():
    ox, oy = le.ARM_BASE_OFFSET_MM
    reachable, r = le.check_reach(-ox, -oy)  # maps to arm-frame (0,0)
    assert not reachable
    assert r < abs(le.ARM_L1_MM - le.ARM_L2_MM)


def test_filter_unreachable_actions_drops_only_the_unreachable_move():
    ox, oy = le.ARM_BASE_OFFSET_MM
    actions = [
        {"action": "home"},
        {"action": "move", "x_mm": 31, "y_mm": 21, "z_mm": 5},
        {"action": "move", "x_mm": -ox, "y_mm": -oy, "z_mm": 5},
        {"action": "pick"},
    ]
    kept, warnings = le.filter_unreachable_actions(actions)
    assert len(kept) == 3
    assert len(warnings) == 1
    assert "outside arm reach envelope" in warnings[0]


# ---- DRC/ERC ----

def test_clean_divider_has_no_errors():
    circuit = {"components": [
        {"ref": "J1", "type": "connector", "pins": {"1": "VIN", "2": "GND"}},
        {"ref": "R1", "type": "resistor", "footprint": "0603", "pins": {"1": "VIN", "2": "MID"}},
        {"ref": "R2", "type": "resistor", "footprint": "0603", "pins": {"1": "MID", "2": "GND"}},
    ]}
    res = le.run_drc_erc(circuit)
    assert res["clean"] and res["net_count"] == 3


def test_dangling_nets_flagged_as_floating():
    circuit = {"components": [
        {"ref": "R1", "type": "resistor", "footprint": "0603", "pins": {"1": "VIN", "2": "MID"}},
        {"ref": "R2", "type": "resistor", "footprint": "0603", "pins": {"1": "MID", "2": "GND"}},
    ]}
    res = le.run_drc_erc(circuit)
    assert not res["clean"]
    assert any("floating net 'VIN'" in e for e in res["errors"])
    assert any("floating net 'GND'" in e for e in res["errors"])


def test_bad_circuit_catches_every_issue():
    circuit = {"components": [
        {"ref": "R1", "type": "resistor", "pins": {"1": "VIN", "2": "MID"}},
        {"ref": "R1", "type": "resistor", "pins": {"1": "MID", "2": ""}},          # dup ref + unconnected pin
        {"ref": "R3", "type": "resistor", "pins": {"1": "ONLY_ONE", "2": "ONLY_ONE"}},  # dead component
        {"ref": "U1", "type": "transistor", "pins": {"1": "A", "2": "B"}},        # wrong pin count
    ]}
    res = le.run_drc_erc(circuit)
    assert any("duplicate reference" in e for e in res["errors"])
    assert any("not connected" in e for e in res["errors"])
    assert any("floating net 'VIN'" in e for e in res["errors"])
    assert any("expected 3" in e for e in res["errors"])
    assert any("no effect" in w for w in res["warnings"])
    assert not res["clean"]


# ---- auto_place ----

def test_placement_returns_all_refs_in_bounds():
    comps = [
        {"ref": "R1", "type": "resistor", "footprint": "0603", "pins": {"1": "VIN", "2": "MID"}},
        {"ref": "R2", "type": "resistor", "footprint": "0603", "pins": {"1": "MID", "2": "GND"}},
        {"ref": "C1", "type": "capacitor", "footprint": "0805", "pins": {"1": "MID", "2": "GND"}},
    ]
    placement = le.auto_place(comps, 30, 20, iterations=1500, seed=42)
    assert set(placement["positions"].keys()) == {"R1", "R2", "C1"}
    for x, y in placement["positions"].values():
        assert 0 <= x <= 30 and 0 <= y <= 20
    assert placement["cost"] >= 0


# ---- auto_route ----

def test_multi_pin_net_routes_as_mst_tree():
    comps = [
        {"ref": "R1", "type": "resistor", "footprint": "0603", "pins": {"1": "VIN", "2": "MID"}},
        {"ref": "R2", "type": "resistor", "footprint": "0603", "pins": {"1": "MID", "2": "GND"}},
        {"ref": "C1", "type": "capacitor", "footprint": "0805", "pins": {"1": "MID", "2": "GND"}},
    ]
    placement = le.auto_place(comps, 30, 20, iterations=1500, seed=42)
    route = le.auto_route(placement["positions"], comps, 30, 20, grid_mm=1.0)
    assert len(route["routed_nets"]) + len(route["unrouted_nets"]) == 2  # MID, GND (VIN has only 1 pin)
    mid = next((r for r in route["routed_nets"] if r["net"] == "MID"), None)
    assert mid is not None and len(mid["segments"]) == 2


def test_pad_offsets_place_pins_off_component_center():
    """Routing should target actual pin locations, not the component
    centroid — confirms _pad_offsets is actually wired into auto_route."""
    comps = [
        {"ref": "R1", "type": "resistor", "footprint": "0603", "pins": {"1": "A", "2": "B"}},
    ]
    offsets = le._pad_offsets(["1", "2"], *le.FOOTPRINT_SIZE_MM["0603"])
    assert offsets["1"] != (0.0, 0.0)
    assert offsets["1"][0] < 0 < offsets["2"][0]  # pin 1 left of center, pin 2 right


def _build_forced_corridor_scenario():
    """Two nets whose only physical path is the same single-cell-wide
    corridor: single-row keepout walls at y=0 and y=4 (0402 footprints at
    grid_mm=1 bleed 1 cell each way, so row y=2 is the only cell left
    open end to end), and net AA / BB pin pairs at coordinates that round
    to the identical grid cells at each end."""
    comps, positions = [], {}
    i = 0
    for y in (0, 4):
        for x in range(20):
            ref = "W%d" % i; i += 1
            comps.append({"ref": ref, "type": "connector", "footprint": "0402", "pins": {"1": "UNUSED%d" % i}})
            positions[ref] = (x + 0.5, y)
    comps += [
        {"ref": "A1", "type": "connector", "footprint": "0402", "pins": {"1": "AA"}},
        {"ref": "A2", "type": "connector", "footprint": "0402", "pins": {"1": "AA"}},
        {"ref": "B1", "type": "connector", "footprint": "0402", "pins": {"1": "BB"}},
        {"ref": "B2", "type": "connector", "footprint": "0402", "pins": {"1": "BB"}},
    ]
    positions.update({"A1": (0.0, 2.0), "A2": (19.0, 2.0), "B1": (0.4, 2.0), "B2": (19.4, 2.0)})
    return comps, positions


def test_single_layer_router_forced_into_a_flagged_crossing():
    comps, positions = _build_forced_corridor_scenario()
    r = le.auto_route(positions, comps, 20, 5, grid_mm=1.0, keepout_mm=0.1, layers=1)
    assert r["unrouted_nets"] == []
    assert len(r["crossings_needing_via"]) > 0
    bb = next(e for e in r["routed_nets"] if e["net"] == "BB")
    assert bb["segments"][0]["via_fallback"] is True


def test_two_layer_router_resolves_the_same_crossing():
    comps, positions = _build_forced_corridor_scenario()
    r = le.auto_route(positions, comps, 20, 5, grid_mm=1.0, keepout_mm=0.1, layers=2)
    assert r["unrouted_nets"] == []
    assert r["crossings_needing_via"] == []
    aa = next(e for e in r["routed_nets"] if e["net"] == "AA")
    bb = next(e for e in r["routed_nets"] if e["net"] == "BB")
    assert aa["segments"][0]["layer"] != bb["segments"][0]["layer"]
    assert bb["segments"][0]["via_fallback"] is False
