"""FastAPI application for PCB golden-board re-inspection.

Endpoints
---------
``POST /api/v1/inspect``
    Multipart upload of two 512x512 grayscale PNG files (``golden`` and
    ``inspected``). Returns registration / exposure evidence plus the
    defect mask and bounding boxes in *inspected-image* coordinates.
``POST /api/v1/placement``
    Multipart upload of the same two PNGs plus JSON box lists: up to six
    non-overlapping 32x32 reference boxes on the golden board and up to six
    32x32 candidate boxes on the inspected board. Returns a per-reference
    correct / wrong / missing / undeterminable verdict.
``GET  /health``
    Liveness probe.
``GET  /docs``
    Swagger UI with an upload form (Try it out).

A content-level inability to decide is HTTP 200 with
``status="undeterminable"`` (it is a valid inspection outcome). Bad input
is HTTP 422/400.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .inspection.detector import inspect_images
from .inspection.io_utils import (
    IMAGE_SIZE,
    ImageValidationError,
    decode_gray_png,
)
from .inspection.placement import (
    PlacementInputError,
    analyze_placement,
    parse_boxes,
    assert_reference_boxes_disjoint,
)

app = FastAPI(
    title="PCB Golden-Board Re-inspection API",
    version="1.0.0",
    description=(
        "Upload a 512x512 grayscale golden-board PNG and an inspected PNG. "
        "The service estimates a global affine registration and linear "
        "exposure compensation (robust to defects), then returns the defect "
        "mask / boxes in inspected-image coordinates. Ambiguous content "
        "(solid color, periodic texture, untrusted correction) returns "
        "status **undeterminable**, never pass."
    ),
)


class WarpParams(BaseModel):
    tx_px: float
    ty_px: float
    rotation_deg: float
    scale_x: float
    scale_y: float
    shear_deg: float


class Correction(BaseModel):
    warp_matrix: list[list[float]] = Field(
        description="2x3 affine, maps golden coords to inspected coords"
    )
    warp_convention: str
    warp_params: WarpParams
    gain: float = Field(description="Fitted generation gain: I = gain*G + bias")
    bias: float = Field(description="Fitted generation bias")
    exposure_model: str
    correction_gain: float = Field(
        description="Multiplier that maps inspected intensities back to golden"
    )
    correction_bias: float = Field(
        description="Offset that maps inspected intensities back to golden"
    )
    robust_sigma: float
    ecc_correlation: float
    overlap_ratio: float


class InspectResponse(BaseModel):
    status: str = Field(
        description="pass | defect | undeterminable"
    )
    reason_code: str | None = Field(
        default=None,
        description=(
            "Set when undeterminable: SOLID_COLOR, LOW_TEXTURE, "
            "PERIODIC_TEXTURE, ECC_FAILED, LOW_CORRELATION, "
            "IMPLAUSIBLE_WARP, INSUFFICIENT_OVERLAP, LOW_INLIER_RATIO, "
            "EXPOSURE_FIT_FAILED, IMPLAUSIBLE_EXPOSURE"
        ),
    )
    reason: str | None = None
    metrics: dict
    correction: Correction | None = None
    defect_mask_png_base64: str | None = Field(
        default=None,
        description="PNG-encoded binary defect mask, inspected coordinates",
    )
    defect_boxes: list[list[int]] = Field(
        default_factory=list,
        description="[x, y, w, h] bounding boxes in inspected coordinates",
    )
    debug_png_base64: str | None = Field(
        default=None, description="2x2 diagnostic panel (PNG, base64)"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "image_size": IMAGE_SIZE}


class PlacementScores(BaseModel):
    ncc: float | None = None
    mae: float | None = None
    ncc_margin_to_runner_up: float | None = None
    mae_margin_to_runner_up: float | None = None
    center_distance_px: float | None = None
    nearest_candidate_distance_px: float | None = None


class PlacementResult(BaseModel):
    reference_index: int
    reference_box: list[int] = Field(description="[x, y, w, h] on the golden image")
    candidate_box: list[int] | None = Field(
        default=None, description="[x, y, w, h] on the inspected image, when assigned"
    )
    candidate_index: int | None = None
    verdict: str = Field(
        description="correct | wrong | missing | undeterminable"
    )
    reason_code: str | None = None
    reason: str | None = None
    scores: PlacementScores | None = None


class UnassignedCandidate(BaseModel):
    candidate_index: int
    candidate_box: list[int]


class PlacementResponse(BaseModel):
    status: str = Field(
        description=(
            "ok when registration/exposure are trusted and per-reference "
            "verdicts follow; undeterminable when the whole board is refused"
        )
    )
    reason_code: str | None = Field(
        default=None,
        description=(
            "Whole-board refusal codes mirror the inspection pipeline: "
            "SOLID_COLOR, LOW_TEXTURE, PERIODIC_TEXTURE, ECC_FAILED, "
            "LOW_CORRELATION, IMPLAUSIBLE_WARP, INSUFFICIENT_OVERLAP, "
            "LOW_INLIER_RATIO, EXPOSURE_FIT_FAILED, IMPLAUSIBLE_EXPOSURE. "
            "Per-position codes: WRONG_COMPONENT, NO_CANDIDATE_AT_POSITION, "
            "NO_GEOMETRIC_CANDIDATE, IDENTITY_AMBIGUOUS, MATCH_INCONCLUSIVE, "
            "INSUFFICIENT_WINDOW_OVERLAP."
        ),
    )
    reason: str | None = None
    results: list[PlacementResult]
    unassigned_candidates: list[UnassignedCandidate] = Field(default_factory=list)
    metrics: dict


@app.post("/api/v1/placement", response_model=PlacementResponse)
async def placement(
    golden: UploadFile = File(..., description="512x512 grayscale golden PNG"),
    inspected: UploadFile = File(
        ..., description="512x512 grayscale inspected PNG"
    ),
    reference_boxes: str = Form(
        ...,
        description=(
            "JSON list of up to 6 non-overlapping 32x32 boxes on the golden "
            'image: [{"x":..,"y":..,"w":32,"h":32}, ...]'
        ),
    ),
    candidate_boxes: str = Form(
        ...,
        description=(
            "JSON list of up to 6 32x32 candidate boxes on the inspected "
            'image: [{"x":..,"y":..,"w":32,"h":32}, ...]'
        ),
    ),
) -> dict:
    golden_bytes = await golden.read()
    inspected_bytes = await inspected.read()
    try:
        g = decode_gray_png(golden_bytes)
        i = decode_gray_png(inspected_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        refs_raw = _parse_box_json(reference_boxes, "reference")
        cands_raw = _parse_box_json(candidate_boxes, "candidate")
        refs = parse_boxes(refs_raw, kind="reference")
        cands = parse_boxes(cands_raw, kind="candidate", allow_empty=True)
        assert_reference_boxes_disjoint(refs)
    except PlacementInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return analyze_placement(g, i, refs, cands)


def _parse_box_json(text: str, kind: str) -> list[dict]:
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PlacementInputError(
            f"{kind}_boxes must be a JSON list of box objects"
        ) from exc
    if not isinstance(data, list):
        raise PlacementInputError(f"{kind}_boxes must be a JSON list")
    return data


@app.post("/api/v1/inspect", response_model=InspectResponse)
async def inspect(
    golden: UploadFile = File(..., description="512x512 grayscale golden PNG"),
    inspected: UploadFile = File(
        ..., description="512x512 grayscale inspected PNG"
    ),
    include_debug: bool = Form(
        default=True, description="Include the 2x2 diagnostic panel"
    ),
) -> dict:
    golden_bytes = await golden.read()
    inspected_bytes = await inspected.read()
    try:
        g = decode_gray_png(golden_bytes)
        i = decode_gray_png(inspected_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return inspect_images(g, i, include_debug=include_debug)
