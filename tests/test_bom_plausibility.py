import layout_engine as le


def test_plausible_bom_has_no_warnings():
    circuit = {"components": [
        {"ref": "R1", "type": "resistor", "value": "10k", "footprint": "0603", "pins": {"1": "A", "2": "B"}},
        {"ref": "C1", "type": "capacitor", "value": "100nF", "footprint": "0603", "pins": {"1": "A", "2": "B"}},
        {"ref": "L1", "type": "inductor", "value": "10uH", "footprint": "0805", "pins": {"1": "A", "2": "B"}},
    ]}
    res = le.check_bom_plausibility(circuit)
    assert res["plausible"] is True
    assert res["warnings"] == []


def test_capacitor_value_on_a_resistor_is_flagged():
    circuit = {"components": [
        {"ref": "R1", "type": "resistor", "value": "100nF", "pins": {"1": "A", "2": "B"}},
    ]}
    res = le.check_bom_plausibility(circuit)
    assert not res["plausible"]
    assert any("doesn't look like a resistance" in w for w in res["warnings"])


def test_missing_value_on_passive_is_flagged():
    circuit = {"components": [
        {"ref": "C1", "type": "capacitor", "pins": {"1": "A", "2": "B"}},
    ]}
    res = le.check_bom_plausibility(circuit)
    assert any("no value specified" in w for w in res["warnings"])


def test_unrecognized_footprint_is_flagged():
    circuit = {"components": [
        {"ref": "U1", "type": "ic", "footprint": "made-up-package-99", "pins": {"1": "A", "2": "B"}},
    ]}
    res = le.check_bom_plausibility(circuit)
    assert any("isn't one this codebase recognizes" in w for w in res["warnings"])


def test_ic_with_no_value_check_is_not_flagged():
    """ICs/connectors don't have a resistance/capacitance/inductance value
    format to validate - only the passives with a _VALUE_CHECKS entry do."""
    circuit = {"components": [
        {"ref": "U1", "type": "ic", "footprint": "soic-8", "pins": {"1": "A", "2": "B"}},
    ]}
    res = le.check_bom_plausibility(circuit)
    assert res["plausible"] is True


def test_various_valid_resistor_formats_accepted():
    for value in ["10k", "4.7k", "220", "1M", "100R", "2.2meg"]:
        circuit = {"components": [{"ref": "R1", "type": "resistor", "value": value, "pins": {"1": "A", "2": "B"}}]}
        res = le.check_bom_plausibility(circuit)
        assert res["plausible"], (value, res["warnings"])
