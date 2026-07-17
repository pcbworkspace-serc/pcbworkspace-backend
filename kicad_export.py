"""
kicad_export.py — serializes the internal circuit JSON (as returned by
/schematic/generate) into KiCad's legacy netlist format ("format D") — the
same format eeschema's "Generate Netlist" produces and pcbnew's "Read
Netlist" consumes to place footprints and build the ratsnest.

Before this, /schematic/generate's output was JSON that's useful for DRC
but isn't a file you can hand to real EDA tooling. This is that missing
link: real components -> real footprints -> a file KiCad can import.

Honesty check: this implements the documented minimal-viable subset of
the format (design/components/nets — the parts pcbnew's netlist reader
needs) and is tested for syntactic well-formedness and structural
correctness against its own output (see tests/test_kicad_export.py). It
has NOT been round-tripped through a real KiCad install — there isn't one
in this environment. Test an actual import before relying on it.
"""

import time

# Common footprint codes (matching layout_engine.FOOTPRINT_SIZE_MM) mapped
# to real KiCad standard library footprint names. Anything not in this
# table passes through as-is, which usually means "wrong library path,
# fix manually in KiCad" rather than "footprint missing" — pcbnew will
# still import the net, just without a resolved footprint for that part.
FOOTPRINT_MAP = {
    "0402": "Resistor_SMD:R_0402_1005Metric",
    "0603": "Resistor_SMD:R_0603_1608Metric",
    "0805": "Resistor_SMD:R_0805_2012Metric",
    "sot-23": "Package_TO_SOT_SMD:SOT-23",
    "sot-223": "Package_TO_SOT_SMD:SOT-223-3_TabPin2",
    "soic-8": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
    "qfp-32": "Package_QFP:LQFP-32_7x7mm_P0.8mm",
    "dip-8": "Package_DIP:DIP-8_W7.62mm",
}


def _esc(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def to_kicad_netlist(circuit, source_name="Layla-generated", tool_name="PCBWorkspace/Layla"):
    """circuit = {"components": [{"ref","type","value","footprint","pins": {pin_no: net_name}}]}
    Returns the netlist as a KiCad legacy-format string."""
    components = circuit.get("components") or []

    net_members = {}  # net_name -> [(ref, pin_no), ...]
    for comp in components:
        ref = comp.get("ref")
        for pin_no, net in (comp.get("pins") or {}).items():
            if net:
                net_members.setdefault(str(net), []).append((ref, str(pin_no)))

    lines = [
        '(export (version "D")',
        '  (design',
        '    (source "%s")' % _esc(source_name),
        '    (date "%s")' % _esc(time.strftime("%Y-%m-%dT%H:%M:%S")),
        '    (tool "%s"))' % _esc(tool_name),
        '  (components',
    ]
    for comp in components:
        ref = comp.get("ref", "")
        value = comp.get("value") or comp.get("type") or ""
        footprint_key = (comp.get("footprint") or "").strip().lower()
        footprint = FOOTPRINT_MAP.get(footprint_key, comp.get("footprint") or "")
        lines.append('    (comp (ref "%s")' % _esc(ref))
        lines.append('      (value "%s")' % _esc(value))
        lines.append('      (footprint "%s"))' % _esc(footprint))
    lines.append('  )')

    lines.append('  (nets')
    for i, (net_name, members) in enumerate(sorted(net_members.items()), start=1):
        lines.append('    (net (code "%d") (name "%s")' % (i, _esc(net_name)))
        for ref, pin_no in members:
            lines.append('      (node (ref "%s") (pin "%s"))' % (_esc(ref), _esc(pin_no)))
        lines.append('    )')
    lines.append('  ))')

    return "\n".join(lines)
