from PIL import Image

import data_capture as dc


def _img():
    return Image.new("RGB", (64, 64), color=(10, 20, 30))


def test_save_capture_writes_image_and_sidecar(tmp_path):
    base = str(tmp_path)
    capture_id = dc.save_capture("align", _img(), {"delta_theta_deg": 5.0}, base_dir=base)
    d = tmp_path / "align"
    assert (d / f"{capture_id}.png").exists()
    sidecar = d / f"{capture_id}.json"
    assert sidecar.exists()
    import json
    data = json.loads(sidecar.read_text())
    assert data["ground_truth"] is None
    assert data["model_prediction"]["delta_theta_deg"] == 5.0


def test_record_feedback_attaches_ground_truth(tmp_path):
    base = str(tmp_path)
    capture_id = dc.save_capture("validate", _img(), {"decision": "PASS"}, base_dir=base)

    updated = dc.record_feedback("validate", capture_id, {"decision": "FAIL", "reason": "tombstoned"}, base_dir=base)
    assert updated["ground_truth"]["decision"] == "FAIL"
    assert updated["labeled_at"] is not None


def test_record_feedback_missing_id_returns_none(tmp_path):
    assert dc.record_feedback("align", "does-not-exist", {"x": 1}, base_dir=str(tmp_path)) is None


def test_load_labeled_dataset_only_returns_labeled(tmp_path):
    base = str(tmp_path)
    labeled_id = dc.save_capture("align", _img(), {"delta_theta_deg": 1.0}, base_dir=base)
    dc.save_capture("align", _img(), {"delta_theta_deg": 2.0}, base_dir=base)  # left unlabeled
    dc.record_feedback("align", labeled_id, {"delta_theta_deg": 1.2}, base_dir=base)

    dataset = dc.load_labeled_dataset("align", base_dir=base)
    assert len(dataset) == 1
    assert dataset[0]["capture_id"] == labeled_id
    assert "image_path" in dataset[0]


def test_capture_stats(tmp_path):
    base = str(tmp_path)
    a = dc.save_capture("align", _img(), {}, base_dir=base)
    dc.save_capture("align", _img(), {}, base_dir=base)
    dc.record_feedback("align", a, {}, base_dir=base)

    stats = dc.capture_stats(base_dir=base)
    assert stats["align"] == {"total_captured": 2, "labeled": 1, "unlabeled": 1}
    assert stats["validate"] == {"total_captured": 0, "labeled": 0, "unlabeled": 0}


def test_save_capture_records_board_session_and_context(tmp_path):
    base = str(tmp_path)
    cid = dc.save_capture("detect", _img(), {"boxes": []}, base_dir=base,
                          board_id=" B1 ", session_id="S1", context={"px_per_mm": 12.5})
    import json
    data = json.loads((tmp_path / "detect" / f"{cid}.json").read_text())
    assert (data["board_id"], data["session_id"]) == ("B1", "S1")
    assert data["capture_context"] == {"px_per_mm": 12.5}
    assert data["label_source"] is None and data["labeler"] is None


def test_record_feedback_records_who_labeled(tmp_path):
    base = str(tmp_path)
    cid = dc.save_capture("align", _img(), {}, base_dir=base)
    updated = dc.record_feedback("align", cid, {"delta_theta_deg": 1.0}, base_dir=base,
                                 label_source="human", labeler="jd", board_id="B9", session_id="S2")
    assert (updated["label_source"], updated["labeler"]) == ("human", "jd")
    assert (updated["board_id"], updated["session_id"]) == ("B9", "S2")


def test_record_feedback_rejects_bad_provenance(tmp_path):
    import pytest
    base = str(tmp_path)
    cid = dc.save_capture("align", _img(), {}, base_dir=base, board_id="B1")
    with pytest.raises(ValueError, match="labeler"):
        dc.record_feedback("align", cid, {}, base_dir=base, label_source="human")
    with pytest.raises(ValueError, match="label_source"):
        dc.record_feedback("align", cid, {}, base_dir=base, label_source="intern")
    with pytest.raises(ValueError, match="conflicts"):
        dc.record_feedback("align", cid, {}, base_dir=base, label_source="model", board_id="B2")


def test_load_labeled_dataset_does_not_create_folders(tmp_path):
    assert dc.load_labeled_dataset("detect", base_dir=str(tmp_path)) == []
    assert not (tmp_path / "detect").exists()
