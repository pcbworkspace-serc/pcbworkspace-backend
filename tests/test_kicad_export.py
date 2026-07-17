"""kicad_export.py has no real KiCad install to round-trip against in this
environment, so these tests do the next best thing: parse the output with
a minimal generic S-expression parser (independent of kicad_export.py's
own logic) and check the resulting structure directly. That verifies
syntactic well-formedness and catches paren-balance bugs it wouldn't be
safe to just eyeball."""
import kicad_export as ke


def _tokenize(text):
    return text.replace("(", " ( ").replace(")", " ) ").split()


def _parse_sexpr(text):
    """Tiny recursive-descent S-expression parser. Strings stay quoted
    (good enough to check paren structure and pull out ref/value/name
    tokens without needing a full KiCad-grammar-aware parser)."""
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in "()":
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                j += 1
            tokens.append(text[i:j + 1])
            i = j + 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < len(text) and not text[j].isspace() and text[j] not in "()":
                j += 1
            tokens.append(text[i:j])
            i = j

    def parse(pos):
        assert tokens[pos] == "("
        node = []
        pos += 1
        while tokens[pos] != ")":
            if tokens[pos] == "(":
                sub, pos = parse(pos)
                node.append(sub)
            else:
                node.append(tokens[pos])
                pos += 1
        return node, pos + 1

    node, end_pos = parse(0)
    assert end_pos == len(tokens), "trailing tokens after top-level expression - unbalanced parens"
    return node


def _find(node, tag):
    """First direct child list whose first element equals tag."""
    return next((c for c in node if isinstance(c, list) and c and c[0] == tag), None)


def _find_all(node, tag):
    return [c for c in node if isinstance(c, list) and c and c[0] == tag]


CIRCUIT = {"components": [
    {"ref": "J1", "type": "connector", "value": "conn", "pins": {"1": "VIN", "2": "GND"}},
    {"ref": "R1", "type": "resistor", "value": "10k", "footprint": "0603", "pins": {"1": "VIN", "2": "MID"}},
    {"ref": "R2", "type": "resistor", "value": "10k", "footprint": "0603", "pins": {"1": "MID", "2": "GND"}},
]}


def test_output_is_well_formed_sexpr():
    text = ke.to_kicad_netlist(CIRCUIT)
    root = _parse_sexpr(text)  # raises if parens don't balance
    assert root[0] == "export"


def test_all_components_and_footprints_present():
    text = ke.to_kicad_netlist(CIRCUIT)
    root = _parse_sexpr(text)
    components = _find(root, "components")
    comps = _find_all(components, "comp")
    assert len(comps) == 3

    refs = {c[1][1].strip('"') for c in comps}  # (comp (ref "R1") ...) -> c[1] == ["ref", '"R1"']
    assert refs == {"J1", "R1", "R2"}

    r1 = next(c for c in comps if c[1][1].strip('"') == "R1")
    footprint_node = _find(r1, "footprint")
    assert footprint_node[1].strip('"') == "Resistor_SMD:R_0603_1608Metric"


def test_all_nets_and_nodes_present():
    text = ke.to_kicad_netlist(CIRCUIT)
    root = _parse_sexpr(text)
    nets = _find(root, "nets")
    net_list = _find_all(nets, "net")
    assert len(net_list) == 3  # VIN, MID, GND

    names = set()
    for net in net_list:
        name_node = _find(net, "name")
        names.add(name_node[1].strip('"'))
    assert names == {"VIN", "MID", "GND"}

    mid_net = next(n for n in net_list if _find(n, "name")[1].strip('"') == "MID")
    nodes = _find_all(mid_net, "node")
    assert len(nodes) == 2  # R1 pin 2, R2 pin 1


def test_unmapped_footprint_passes_through():
    circuit = {"components": [
        {"ref": "U1", "type": "ic", "footprint": "custom-24pin", "pins": {"1": "A", "2": "B"}},
    ]}
    text = ke.to_kicad_netlist(circuit)
    root = _parse_sexpr(text)
    comp = _find_all(_find(root, "components"), "comp")[0]
    assert _find(comp, "footprint")[1].strip('"') == "custom-24pin"


def test_special_characters_in_ref_are_escaped_safely():
    circuit = {"components": [
        {"ref": 'R"1', "type": "resistor", "pins": {"1": "A", "2": "B"}},
    ]}
    text = ke.to_kicad_netlist(circuit)
    # must still parse cleanly even with an escaped quote inside a ref
    root = _parse_sexpr(text)
    assert root[0] == "export"
