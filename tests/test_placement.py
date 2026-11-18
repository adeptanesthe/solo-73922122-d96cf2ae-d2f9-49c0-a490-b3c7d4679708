"""Functional tests for the placement verification endpoint and engine.

Covers the contract required by the placement feature:

* correct boards return ``correct`` per reference;
* a two-part swap reports BOTH positions as ``wrong`` simultaneously;
* a three-way rotation reports all moved parts ``wrong``;
* a missing part (no covering candidate) reports ``missing`` while the
  other, decidable positions still get firm verdicts;
* byte-identical and near-look-alike components yield ``undeterminable``
  only at the affected positions -- neighbouring distinct components are
  still judged;
* flat, low-texture components are still separated by the intensity-level
  (MAE) channel, so a swap of two different-gray flat parts is detected;
* an untrusted registration/exposure refuses the whole board, never
  reporting correct or missing;
* box validation (size, count, overlap, frame bounds) returns HTTP 400;
* a candidate can never be assigned to two reference positions.
"""
from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inspection.placement import analyze_placement
from app.main import app
from tests.synth.placement_cases import (
    BOX,
    make_correct_scenario,
    make_duplicate_scenario,
    make_flat_parts_scenario,
    make_lookalike_scenario,
    make_missing_scenario,
    make_solid_board_scenario,
    make_swap_scenario,
    make_three_way_scenario,
)

client = TestClient(app)


def _run_engine(scenario):
    from app.inspection.placement import Box
    refs = [Box(*b) for b in scenario.reference_boxes]
    cands = [Box(*b) for b in scenario.candidate_boxes]
    return analyze_placement(scenario.golden, scenario.inspected, refs, cands)


def _post_api(scenario):
    files = {
        "golden": ("golden.png", io.BytesIO(_png(scenario.golden)), "image/png"),
        "inspected": ("inspected.png", io.BytesIO(_png(scenario.inspected)),
                      "image/png"),
    }
    data = {
        "reference_boxes": json.dumps([
            {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
            for b in scenario.reference_boxes
        ]),
        "candidate_boxes": json.dumps([
            {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
            for b in scenario.candidate_boxes
        ]),
    }
    return client.post("/api/v1/placement", files=files, data=data)


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _verdicts(out: dict) -> dict[int, str]:
    return {r["reference_index"]: r["verdict"] for r in out["results"]}


def _assigned_candidates(out: dict) -> list[int]:
    return [r["candidate_index"] for r in out["results"]
            if r["candidate_index"] is not None]


# ---------------------------------------------------------------- engine

def test_correct_board_all_positions_correct():
    scenario = make_correct_scenario()
    out = _run_engine(scenario)
    assert out["status"] == "ok"
    assert _verdicts(out) == scenario.expected
    for row in out["results"]:
        assert row["scores"]["ncc"] >= 0.75
        assert row["scores"]["mae"] <= 12.0


def test_swap_reports_both_positions_wrong():
    scenario = make_swap_scenario()
    out = _run_engine(scenario)
    assert out["status"] == "ok"
    verdicts = _verdicts(out)
    assert verdicts == scenario.expected
    # Both swapped positions must be wrong at the same time.
    assert verdicts[0] == "wrong" and verdicts[1] == "wrong"
    swapped = [r for r in out["results"] if r["reference_index"] in (0, 1)]
    assert all(r["reason_code"] == "WRONG_COMPONENT" for r in swapped)
    # And each candidate is used at most once.
    assert len(_assigned_candidates(out)) == len(set(_assigned_candidates(out)))


def test_three_way_rotation_all_moved_parts_wrong():
    scenario = make_three_way_scenario()
    out = _run_engine(scenario)
    assert out["status"] == "ok"
    assert _verdicts(out) == scenario.expected


def test_missing_part_only_affected_position_refused_or_missing():
    scenario = make_missing_scenario()
    out = _run_engine(scenario)
    assert out["status"] == "ok"
    verdicts = _verdicts(out)
    # The empty footprint must be declared missing; all other positions stay
    # decidable (correct), never silently refused.
    assert verdicts[2] == "missing"
    for idx in (0, 1, 3):
        assert verdicts[idx] == "correct", (idx, verdicts)
    missing_row = next(r for r in out["results"] if r["reference_index"] == 2)
    assert missing_row["candidate_box"] is None
    assert missing_row["reason_code"] == "NO_CANDIDATE_AT_POSITION"


def test_duplicate_components_refuse_only_affected_positions():
    scenario = make_duplicate_scenario()
    out = _run_engine(scenario)
    assert out["status"] == "ok"
    verdicts = _verdicts(out)
    assert verdicts[0] == "undeterminable"
    assert verdicts[1] == "undeterminable"
    # Distinct components at other positions remain fully decidable.
    assert verdicts[2] == "correct"
    assert verdicts[3] == "correct"
    refused = [r for r in out["results"]
               if r["reference_index"] in (0, 1)]
    assert all(r["reason_code"] == "IDENTITY_AMBIGUOUS" for r in refused)


def test_lookalike_components_refuse_only_affected_positions():
    # Near-lookalike parts: the identity margin is too small to distinguish
    # them, so the two affected positions are refused; distinct positions
    # elsewhere still receive firm verdicts.
    scenario = make_lookalike_scenario()
    out = _run_engine(scenario)
    assert out["status"] == "ok"
    verdicts = _verdicts(out)
    assert verdicts[0] == "undeterminable"
    assert verdicts[1] == "undeterminable"
    assert verdicts[2] == "correct"
    assert verdicts[3] == "correct"
    refused = [r for r in out["results"]
               if r["reference_index"] in (0, 1)]
    assert all(r["reason_code"] == "IDENTITY_AMBIGUOUS" for r in refused)
    # The ambiguity must be evidenced by a sub-threshold NCC margin.
    assert all(r["scores"]["ncc_margin_to_runner_up"] < 0.10
               for r in refused)


def test_flat_low_texture_parts_still_detected_via_mae():
    # Flat components have almost no structure: the MAE (intensity-level)
    # channel must still flag a swap of two different-gray flat parts.
    scenario = make_flat_parts_scenario()
    out = _run_engine(scenario)
    assert out["status"] == "ok"
    verdicts = _verdicts(out)
    assert verdicts == scenario.expected
    flat_rows = [r for r in out["results"]
                 if r["reference_index"] in (0, 1)]
    assert all(r["reason_code"] == "WRONG_COMPONENT" for r in flat_rows)
    # MAE carries the decision: far above the reject bound, NCC near zero.
    assert all(r["scores"]["mae"] >= 25.0 for r in flat_rows)
    assert all(abs(r["scores"]["ncc"]) < 0.35 for r in flat_rows)


def test_untrusted_board_is_whole_board_refusal():
    scenario = make_solid_board_scenario()
    out = _run_engine(scenario)
    assert out["status"] == "undeterminable"
    assert out["reason_code"] == "SOLID_COLOR"
    # No position may be reported correct or missing.
    assert all(r["verdict"] == "undeterminable" for r in out["results"])
    assert all(r["candidate_index"] is None for r in out["results"])


def test_periodic_board_is_whole_board_refusal():
    # A periodic texture makes registration non-unique: every position must
    # be refused, never guessed as correct or missing.
    from tests.synth.generator import make_periodic_sample
    from app.inspection.placement import Box

    sample = make_periodic_sample(0, checker=False)
    refs = [Box(60, 60, BOX, BOX), Box(200, 200, BOX, BOX)]
    cands = [Box(76, 60, BOX, BOX), Box(216, 200, BOX, BOX)]
    out = analyze_placement(sample.golden, sample.inspected, refs, cands)
    assert out["status"] == "undeterminable"
    assert out["reason_code"] == "PERIODIC_TEXTURE"
    assert len(out["results"]) == 2
    assert all(r["verdict"] == "undeterminable" for r in out["results"])


def test_candidate_never_assigned_to_two_positions():
    scenario = make_correct_scenario(n_components=4)
    # Drop two candidates: the remaining ones must still map one-to-one.
    scenario.candidate_boxes = scenario.candidate_boxes[:2]
    out = _run_engine(scenario)
    used = _assigned_candidates(out)
    assert len(used) == len(set(used))
    assert len(used) == 2


def test_extra_spurious_candidate_is_left_unassigned():
    scenario = make_correct_scenario(n_components=3)
    scenario.candidate_boxes.append([400, 400, BOX, BOX])
    out = _run_engine(scenario)
    assert out["unassigned_candidates"][0]["candidate_index"] == 3
    used = _assigned_candidates(out)
    assert 3 not in used


def test_empty_candidate_list_all_footprints_missing():
    # Zero candidate boxes is allowed: every marked position whose footprint
    # is foreign must be reported missing.
    from tests.synth.placement_cases import (
        _background,
        _build_golden,
        _finalize,
        _rng,
        _random_warp,
        REFERENCE_POSITIONS,
    )
    rng = _rng(900)
    n = 3
    golden, _ = _build_golden(rng, n)
    assembled = golden.copy()
    plain_bg = _background(np.random.default_rng(20261118 + 901))
    for x, y in REFERENCE_POSITIONS[:n]:  # remove every component pre-warp
        assembled[y:y + BOX, x:x + BOX] = plain_bg[y:y + BOX, x:x + BOX]
    warp, _, _, _ = _random_warp(rng)
    scenario = _finalize(
        "placement_all_missing", golden, assembled, warp, rng,
        candidate_positions=[],
        expected={k: "missing" for k in range(n)},
        gain=1.05, bias=-4.0,
    )
    scenario.candidate_boxes = []
    out = _run_engine(scenario)
    assert out["status"] == "ok"
    assert len(out["results"]) == n
    assert all(r["verdict"] == "missing" for r in out["results"])
    assert all(r["candidate_index"] is None for r in out["results"])
    assert out["unassigned_candidates"] == []


def test_low_overlap_edge_window_is_refused_not_guessed():
    scenario = make_correct_scenario(n_components=2)
    # Translate inspected content 22px right: the far-right golden window has
    # no corresponding coverage and must be refused rather than misjudged.
    shifted = cv2.warpAffine(
        scenario.inspected,
        np.array([[1.0, 0.0, 22.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (512, 512), borderValue=0,
    )
    scenario.inspected = shifted
    scenario.reference_boxes = scenario.reference_boxes + [[480, 480, BOX, BOX]]
    scenario.candidate_boxes = scenario.candidate_boxes + [[480, 480, BOX, BOX]]
    out = _run_engine(scenario)
    edge = next(r for r in out["results"] if r["reference_index"] == 2)
    assert edge["verdict"] == "undeterminable"
    assert edge["reason_code"] == "INSUFFICIENT_WINDOW_OVERLAP"
    # The in-frame positions are still judged normally.
    assert _verdicts(out)[0] == "correct"


# ------------------------------------------------------------- validation

def test_rejects_non_32_box():
    scenario = make_correct_scenario(n_components=2)
    bad = [{"x": 10, "y": 10, "w": 16, "h": 16}]
    resp = client.post(
        "/api/v1/placement",
        files={
            "golden": ("g.png", io.BytesIO(_png(scenario.golden)), "image/png"),
            "inspected": ("i.png", io.BytesIO(_png(scenario.inspected)),
                          "image/png"),
        },
        data={
            "reference_boxes": json.dumps(bad),
            "candidate_boxes": json.dumps([
                {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                for b in scenario.candidate_boxes[:1]
            ]),
        },
    )
    assert resp.status_code == 400
    assert "32x32" in resp.json()["detail"]


def test_rejects_overlapping_reference_boxes():
    scenario = make_correct_scenario(n_components=2)
    overlap = [
        {"x": 80, "y": 80, "w": BOX, "h": BOX},
        {"x": 90, "y": 90, "w": BOX, "h": BOX},
    ]
    resp = client.post(
        "/api/v1/placement",
        files={
            "golden": ("g.png", io.BytesIO(_png(scenario.golden)), "image/png"),
            "inspected": ("i.png", io.BytesIO(_png(scenario.inspected)),
                          "image/png"),
        },
        data={
            "reference_boxes": json.dumps(overlap),
            "candidate_boxes": json.dumps([
                {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                for b in scenario.candidate_boxes[:2]
            ]),
        },
    )
    assert resp.status_code == 400
    assert "overlap" in resp.json()["detail"]


def test_rejects_more_than_six_boxes():
    scenario = make_correct_scenario(n_components=2)
    seven = [{"x": 10 + k * 40, "y": 10, "w": BOX, "h": BOX} for k in range(7)]
    resp = client.post(
        "/api/v1/placement",
        files={
            "golden": ("g.png", io.BytesIO(_png(scenario.golden)), "image/png"),
            "inspected": ("i.png", io.BytesIO(_png(scenario.inspected)),
                          "image/png"),
        },
        data={
            "reference_boxes": json.dumps(seven),
            "candidate_boxes": json.dumps([
                {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                for b in scenario.candidate_boxes[:2]
            ]),
        },
    )
    assert resp.status_code == 400
    assert "at most 6" in resp.json()["detail"]


def test_rejects_box_outside_frame():
    scenario = make_correct_scenario(n_components=2)
    bad = [{"x": 500, "y": 500, "w": BOX, "h": BOX}]
    resp = client.post(
        "/api/v1/placement",
        files={
            "golden": ("g.png", io.BytesIO(_png(scenario.golden)), "image/png"),
            "inspected": ("i.png", io.BytesIO(_png(scenario.inspected)),
                          "image/png"),
        },
        data={
            "reference_boxes": json.dumps(bad),
            "candidate_boxes": json.dumps([
                {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                for b in scenario.candidate_boxes[:1]
            ]),
        },
    )
    assert resp.status_code == 400


def test_rejects_malformed_box_json():
    scenario = make_correct_scenario(n_components=2)
    resp = client.post(
        "/api/v1/placement",
        files={
            "golden": ("g.png", io.BytesIO(_png(scenario.golden)), "image/png"),
            "inspected": ("i.png", io.BytesIO(_png(scenario.inspected)),
                          "image/png"),
        },
        data={
            "reference_boxes": "not-json",
            "candidate_boxes": json.dumps([
                {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                for b in scenario.candidate_boxes[:2]
            ]),
        },
    )
    assert resp.status_code == 400


def test_rejects_wrong_size_image():
    small = np.zeros((256, 256), np.uint8)
    box = json.dumps([{"x": 10, "y": 10, "w": BOX, "h": BOX}])
    resp = client.post(
        "/api/v1/placement",
        files={
            "golden": ("g.png", io.BytesIO(_png(small)), "image/png"),
            "inspected": ("i.png", io.BytesIO(_png(small)), "image/png"),
        },
        data={"reference_boxes": box, "candidate_boxes": box},
    )
    assert resp.status_code == 400


# ------------------------------------------------------------------- API

def test_placement_endpoint_happy_path_and_docs():
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/api/v1/placement" in spec["paths"]

    scenario = make_swap_scenario()
    resp = _post_api(scenario)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert _verdicts(body)[0] == "wrong"
    # Every entry carries back the assigned candidate box in inspected coords.
    swapped = [r for r in body["results"]
               if r["reference_index"] in (0, 1)]
    assert all(r["candidate_box"] is not None for r in swapped)
    assert all(len(r["candidate_box"]) == 4 for r in swapped)


def test_placement_endpoint_accepts_empty_candidate_list():
    scenario = make_correct_scenario(n_components=2)
    resp = client.post(
        "/api/v1/placement",
        files={
            "golden": ("g.png", io.BytesIO(_png(scenario.golden)), "image/png"),
            "inspected": ("i.png", io.BytesIO(_png(scenario.inspected)),
                          "image/png"),
        },
        data={
            "reference_boxes": json.dumps([
                {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                for b in scenario.reference_boxes
            ]),
            "candidate_boxes": "[]",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["unassigned_candidates"] == []


def test_original_inspect_endpoint_unchanged():
    from tests.synth.generator import make_normal_sample
    from app.inspection.io_utils import encode_png
    sample = make_normal_sample(0, with_defect=True)
    resp = client.post(
        "/api/v1/inspect",
        files={
            "golden": ("g.png", io.BytesIO(encode_png(sample.golden)),
                       "image/png"),
            "inspected": ("i.png", io.BytesIO(encode_png(sample.inspected)),
                          "image/png"),
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "defect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
