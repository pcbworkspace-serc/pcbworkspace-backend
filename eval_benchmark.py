"""
eval_benchmark.py — small internal netlist-generation benchmark for
/schematic/generate, so "Layla is X% accurate" is a real number instead of
"AI tool for beginners."

This intentionally does NOT call the Anthropic API directly — it hits the
already-running Flask server's /schematic/generate endpoint over HTTP, so
the score reflects exactly what a user gets, DRC included. No new
dependency: uses urllib from the standard library.

Usage:
    python flask_server.py &                       # start the server
    ANTHROPIC_API_KEY=sk-... python eval_benchmark.py

Scoring — "connectivity accuracy":
  For each test case, build the ground-truth net partition (which pins are
  on the same net) and the generated one, then compare every pair of pins
  that exist in BOTH circuits: do they agree on same-net-or-not? This is
  naming-invariant (the LLM doesn't have to call a net "VIN" just because
  the ground truth does) but still catches a wrong connection, a merged
  net, or a split net. Plain net-name string matching would fail on any
  circuit where the model picks reasonable-but-different net names, which
  is most of them — that's not a generation error worth penalizing.

This module has no test-runner dependency (no pytest) — run it directly.
"""

import json
import os
import sys
import urllib.request
import urllib.error

SERVER = os.environ.get("PCBWORKSPACE_SERVER", "http://127.0.0.1:5000")

# ─────────────────────────────────────────────────────────────────────────
# Hand-authored test circuits — small, unambiguous, single correct topology
# ─────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "voltage_divider",
        "description": "A resistive voltage divider from a 5V input to ground, "
                        "with a connector J1 for the 5V input and ground, feeding "
                        "the midpoint to a second connector J2 as the divided output.",
        "ground_truth": {"components": [
            {"ref": "J1", "type": "connector", "pins": {"1": "VIN", "2": "GND"}},
            {"ref": "R1", "type": "resistor", "pins": {"1": "VIN", "2": "MID"}},
            {"ref": "R2", "type": "resistor", "pins": {"1": "MID", "2": "GND"}},
            {"ref": "J2", "type": "connector", "pins": {"1": "MID", "2": "GND"}},
        ]},
    },
    {
        "name": "led_current_limiter",
        "description": "A single LED with a current-limiting resistor in series, "
                        "powered from a connector J1 providing VIN and GND.",
        "ground_truth": {"components": [
            {"ref": "J1", "type": "connector", "pins": {"1": "VIN", "2": "GND"}},
            {"ref": "R1", "type": "resistor", "pins": {"1": "VIN", "2": "MID"}},
            {"ref": "D1", "type": "led", "pins": {"1": "MID", "2": "GND"}},
        ]},
    },
    {
        "name": "rc_lowpass_filter",
        "description": "A first-order RC low-pass filter: input connector J1 (VIN, GND) "
                        "into a series resistor, then a capacitor to ground, with the "
                        "filtered output brought out to connector J2.",
        "ground_truth": {"components": [
            {"ref": "J1", "type": "connector", "pins": {"1": "VIN", "2": "GND"}},
            {"ref": "R1", "type": "resistor", "pins": {"1": "VIN", "2": "OUT"}},
            {"ref": "C1", "type": "capacitor", "pins": {"1": "OUT", "2": "GND"}},
            {"ref": "J2", "type": "connector", "pins": {"1": "OUT", "2": "GND"}},
        ]},
    },
    {
        "name": "decoupled_two_pin_load",
        "description": "A generic 2-pin load component (type ic, treat as a black box "
                        "with a VIN and GND pin) powered from connector J1, with a "
                        "0.1uF decoupling capacitor placed directly across VIN and GND "
                        "near the load.",
        "ground_truth": {"components": [
            {"ref": "J1", "type": "connector", "pins": {"1": "VIN", "2": "GND"}},
            {"ref": "C1", "type": "capacitor", "pins": {"1": "VIN", "2": "GND"}},
            {"ref": "U1", "type": "ic", "pins": {"1": "VIN", "2": "GND"}},
        ]},
    },
    {
        "name": "pull_up_switch",
        "description": "A momentary switch to ground with a pull-up resistor to VIN, "
                        "so the switch node reads high when open and low when pressed. "
                        "VIN and GND come from connector J1; the switch node is brought "
                        "out to connector J2 as the digital input signal.",
        "ground_truth": {"components": [
            {"ref": "J1", "type": "connector", "pins": {"1": "VIN", "2": "GND"}},
            {"ref": "R1", "type": "resistor", "pins": {"1": "VIN", "2": "SIG"}},
            {"ref": "SW1", "type": "switch", "pins": {"1": "SIG", "2": "GND"}},
            {"ref": "J2", "type": "connector", "pins": {"1": "SIG", "2": "GND"}},
        ]},
    },
]


# ─────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────

def _net_partition(circuit):
    """{(ref, pin): net_name} for every connected pin in a circuit."""
    pin_to_net = {}
    for comp in circuit.get("components", []):
        ref = comp.get("ref")
        for pin, net in (comp.get("pins") or {}).items():
            if ref and net:
                pin_to_net[(ref, str(pin))] = str(net)
    return pin_to_net


def connectivity_accuracy(ground_truth, generated):
    """Pairwise same-net agreement over pins present in both circuits.
    Returns (accuracy, n_pairs_compared, n_common_pins). accuracy is None
    if fewer than 2 pins are common (nothing meaningful to compare)."""
    gt = _net_partition(ground_truth)
    gen = _net_partition(generated)
    common = sorted(set(gt.keys()) & set(gen.keys()))
    if len(common) < 2:
        return None, 0, len(common)

    agree, total = 0, 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            same_gt = gt[a] == gt[b]
            same_gen = gen[a] == gen[b]
            total += 1
            if same_gt == same_gen:
                agree += 1
    return (agree / total if total else None), total, len(common)


def component_coverage(ground_truth, generated):
    """Fraction of ground-truth (ref, pin) endpoints the model reproduced
    at all (by ref name) — separate from whether it wired them correctly."""
    gt_refs = {c["ref"] for c in ground_truth.get("components", [])}
    gen_refs = {c["ref"] for c in generated.get("components", [])}
    if not gt_refs:
        return None
    return len(gt_refs & gen_refs) / len(gt_refs)


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

def _call_schematic_generate(description, server=SERVER, timeout=60):
    body = json.dumps({"description": description}).encode("utf-8")
    req = urllib.request.Request(
        server.rstrip("/") + "/schematic/generate",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run(server=SERVER, verbose=True):
    results = []
    for case in TEST_CASES:
        try:
            resp = _call_schematic_generate(case["description"], server=server)
        except urllib.error.URLError as e:
            results.append({"name": case["name"], "error": "server unreachable: %s" % e})
            continue

        if not resp.get("ok"):
            results.append({"name": case["name"], "error": resp.get("error", "unknown error")})
            continue

        generated = resp.get("circuit", {})
        acc, n_pairs, n_common = connectivity_accuracy(case["ground_truth"], generated)
        coverage = component_coverage(case["ground_truth"], generated)
        results.append({
            "name": case["name"],
            "connectivity_accuracy": round(acc, 3) if acc is not None else None,
            "pairs_compared": n_pairs,
            "common_pins": n_common,
            "component_coverage": round(coverage, 3) if coverage is not None else None,
            "drc_clean": resp.get("drc", {}).get("clean"),
            "drc_error_count": len(resp.get("drc", {}).get("errors", [])),
        })

    valid = [r for r in results if r.get("connectivity_accuracy") is not None]
    summary = {
        "n_cases": len(TEST_CASES),
        "n_scored": len(valid),
        "n_errors": len(TEST_CASES) - len(valid),
        "mean_connectivity_accuracy": round(sum(r["connectivity_accuracy"] for r in valid) / len(valid), 3) if valid else None,
        "drc_clean_rate": round(sum(1 for r in valid if r.get("drc_clean")) / len(valid), 3) if valid else None,
    }

    if verbose:
        print("=" * 60)
        print(" Layla /schematic/generate - internal benchmark")
        print("=" * 60)
        for r in results:
            if "error" in r:
                print(f"  {r['name']:<24} ERROR: {r['error']}")
            else:
                print(f"  {r['name']:<24} conn_acc={r['connectivity_accuracy']}"
                      f"  drc_clean={r['drc_clean']}  coverage={r['component_coverage']}")
        print("-" * 60)
        print(f"  scored {summary['n_scored']}/{summary['n_cases']}"
              f"  mean_connectivity_accuracy={summary['mean_connectivity_accuracy']}"
              f"  drc_clean_rate={summary['drc_clean_rate']}")
        print("=" * 60)

    return {"results": results, "summary": summary}


if __name__ == "__main__":
    out = run()
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(out, indent=2))
