"""Component placement verification (wrong-part / missing-part detection).

Given a golden board and an inspected board plus up to six non-overlapping
32x32 reference boxes on the golden image and up to six 32x32 candidate
boxes on the inspected image, this module decides, per reference position,
whether the visible component is the golden component (``correct``), a
different component (``wrong``), absent (``missing``) or not safely
decidable (``undeterminable``).

Trust model
-----------
* The exact same registration / exposure machinery as the re-inspection
  pipeline is reused. When that pipeline does not trust its affine warp or
  its exposure fit, the *whole board* is refused (every position reported
  ``undeterminable``) -- an ambiguous correction can never be turned into a
  "correct" or "missing" verdict.
* Identity is scored in *golden-frame geometry*: the inspected image is
  back-rendered with the inverse warp and brought back to golden intensity
  scale, so a golden reference patch and an inspected candidate patch view
  the same physical texture in a canonical 32x32 window.
* Two complementary texture measures are used: normalized cross
  correlation (structure) and exposure-corrected mean absolute intensity
  error (mean level, which still separates flat/low-texture parts).
* Candidates are assigned to reference positions one-to-one by geometry;
  a candidate can never back two positions. A swap therefore surfaces as
  two ``wrong`` verdicts simultaneously.
* When two identities match nearly equally (look-alike parts) or a
  component itself carries too little information, only the affected
  positions are refused; every other position still gets a firm verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import cv2
import numpy as np

from . import geometry
from .detector import inspect_images
from .exposure import ExposureFit, correct_to_reference
from .io_utils import IMAGE_SIZE

BOX_SIZE = 32
MAX_BOXES = 6

# Geometric assignment: a candidate belongs to a reference position only
# when its mapped center lands within this radius of the reference center.
# Components are 32 px wide, so a part still covering its footprint cannot
# drift further than half a diagonal.
MAX_CENTER_DISTANCE_PX = 28.0

# Texture identity thresholds, calibrated on 32x32 windows after warp and
# exposure correction. Windows are index-aligned in golden-frame geometry
# and the MAE channel is computed on Gaussian-blurred (5x5, sigma=1)
# patches so residual sub-pixel misregistration cannot inflate it:
#   correct textured parts show NCC >= ~0.86 and blurred MAE <= ~8.1;
#   different parts show NCC <= ~0.21 or blurred MAE >= ~35;
#   empty footprints show NCC ~ 0 and blurred MAE >= ~45;
#   anything landing in the gap is refused rather than guessed.
NCC_ACCEPT = 0.75
NCC_REJECT = 0.45
MAE_ACCEPT = 12.0
MAE_REJECT = 25.0
# The observed content must match this position's golden component clearly
# better than every *other* golden component, otherwise two parts are
# visually indistinguishable for this view.
NCC_MARGIN = 0.10
MAE_MARGIN = 8.0
# A golden patch with less spread than this has no structure to correlate;
# the mean-level (MAE) channel alone then carries the decision.
MIN_PATCH_STD = 6.0
# Empty footprint: the corrected window at a position with no candidate must
# not resemble the golden component at all for a "missing" verdict.
MISSING_NCC_MAX = 0.35
MISSING_MAE_MIN = 18.0
MAE_BLUR_KERNEL = 5
# A reference window must have at least this fraction of pixels inside the
# warp's valid overlap; border-fill pixels otherwise poison the scores.
MIN_WINDOW_OVERLAP = 0.75


class PlacementInputError(ValueError):
    """Raised when the supplied reference/candidate boxes are invalid."""


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


def parse_boxes(
    raw: list[dict] | None, *, kind: str, allow_empty: bool = False
) -> list[Box]:
    """Validate one list of uploaded boxes."""
    if not raw:
        if allow_empty:
            return []
        raise PlacementInputError(f"at least one {kind} box is required")
    if len(raw) > MAX_BOXES:
        raise PlacementInputError(
            f"at most {MAX_BOXES} {kind} boxes are allowed, got {len(raw)}"
        )
    boxes: list[Box] = []
    for idx, item in enumerate(raw):
        try:
            x, y, w, h = int(item["x"]), int(item["y"]), \
                int(item["w"]), int(item["h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PlacementInputError(
                f"{kind} box {idx} must contain integer x, y, w, h"
            ) from exc
        if w != BOX_SIZE or h != BOX_SIZE:
            raise PlacementInputError(
                f"{kind} box {idx} must be {BOX_SIZE}x{BOX_SIZE}, got {w}x{h}"
            )
        if x < 0 or y < 0 or x + w > IMAGE_SIZE or y + h > IMAGE_SIZE:
            raise PlacementInputError(
                f"{kind} box {idx} {[x, y, w, h]} lies outside the "
                f"{IMAGE_SIZE}x{IMAGE_SIZE} frame"
            )
        boxes.append(Box(x, y, w, h))
    return boxes


def assert_reference_boxes_disjoint(boxes: list[Box]) -> None:
    """Golden reference boxes must not overlap each other."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a.x < b.x + b.w and b.x < a.x + a.w \
                    and a.y < b.y + b.h and b.y < a.y + a.h:
                raise PlacementInputError(
                    f"reference boxes {i} and {j} overlap; reference boxes "
                    "must be mutually non-overlapping"
                )


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross correlation of two equal-size grayscale windows."""
    af, bf = a.astype(np.float32), b.astype(np.float32)
    if af.std() < 1e-6 or bf.std() < 1e-6:
        # No structure in at least one window: fall back to an intensity-level
        # agreement score (1.0 when means match, decaying with the gap).
        return float(np.clip(1.0 - np.abs(af.mean() - bf.mean()) / 40.0,
                             -1.0, 1.0))
    return float(cv2.matchTemplate(af, bf, cv2.TM_CCOEFF_NORMED)[0, 0])


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    """Exposure-corrected MAE on mildly blurred patches.

    A 5x5 Gaussian blur removes the sub-pixel-registration edge penalty that
    otherwise dominates raw MAE on sharp component outlines, while keeping
    mean-level differences (the only signal a flat component offers).
    """
    af = cv2.GaussianBlur(a.astype(np.float32),
                          (MAE_BLUR_KERNEL, MAE_BLUR_KERNEL), 1.0)
    bf = cv2.GaussianBlur(b.astype(np.float32),
                          (MAE_BLUR_KERNEL, MAE_BLUR_KERNEL), 1.0)
    return float(np.mean(np.abs(af - bf)))


def _geometric_assignment(
    distance: np.ndarray, n_refs: int, n_cands: int
) -> tuple[dict[int, int], list[int]]:
    """One-to-one assignment minimizing center distance.

    Returns a mapping ``reference_idx -> candidate_idx`` for pairs closer
    than :data:`MAX_CENTER_DISTANCE_PX`, plus the candidate indices that no
    reference position claims. When fewer candidates than references exist
    (a part may be missing), every subset of references is considered; with
    at most six boxes the exhaustive search stays under 6! * 6 = 4320
    evaluations.
    """
    k = min(n_refs, n_cands)
    if n_refs <= n_cands:
        ref_subsets: tuple[tuple[int, ...], ...] = (tuple(range(n_refs)),)
    else:
        ref_subsets = tuple(combinations(range(n_refs), n_cands))

    best_cost = np.inf
    best_refs: tuple[int, ...] = ()
    best_perm: tuple[int, ...] = ()
    for refs in ref_subsets:
        for perm in permutations(range(n_cands), k):
            cost = sum(float(distance[r, perm[idx]])
                       for idx, r in enumerate(refs))
            if cost < best_cost:
                best_cost, best_refs, best_perm = cost, refs, perm

    assignment: dict[int, int] = {}
    for idx, r in enumerate(best_refs):
        j = best_perm[idx]
        if distance[r, j] <= MAX_CENTER_DISTANCE_PX:
            assignment[r] = j
    assigned = set(assignment.values())
    leftovers = [j for j in range(n_cands) if j not in assigned]
    return assignment, leftovers


def _identity_verdict(
    golden_patch: np.ndarray,
    own_scores: tuple[float, float],
    cross_scores: list[tuple[float, float]],
) -> tuple[str, str]:
    """Decide correct / wrong / undeterminable for one assigned position.

    ``own_scores`` is (NCC, MAE) between the golden component of *this*
    reference position and the corrected content at that position.
    ``cross_scores`` holds the same pair measured against every *other*
    golden component: the margin between the two tells whether two part
    identities are visually indistinguishable here.
    """
    ncc_own, mae_own = own_scores
    low_texture = float(golden_patch.std()) < MIN_PATCH_STD

    runner_ncc = max((s[0] for s in cross_scores), default=-1.0)
    nearest_mae = min((s[1] for s in cross_scores), default=np.inf)
    ncc_margin = ncc_own - runner_ncc
    mae_margin = nearest_mae - mae_own

    ncc_good = ncc_own >= NCC_ACCEPT
    ncc_bad = ncc_own <= NCC_REJECT
    mae_good = mae_own <= MAE_ACCEPT
    mae_bad = mae_own >= MAE_REJECT

    if ncc_good and mae_good:
        if not low_texture and ncc_margin < NCC_MARGIN:
            return ("undeterminable",
                    "IDENTITY_AMBIGUOUS: another golden component matches the "
                    "observed texture almost equally well")
        if low_texture and mae_margin < MAE_MARGIN:
            return ("undeterminable",
                    "IDENTITY_AMBIGUOUS: low-texture part and another golden "
                    "component has a nearly identical intensity level")
        return "correct", "component matches the golden reference"
    if ncc_bad or mae_bad:
        return ("wrong",
                "observed component texture does not match the golden "
                "component for this reference position")

    return ("undeterminable",
            "MATCH_INCONCLUSIVE: similarity is between the accept and reject "
            "bounds, so the component identity cannot be confirmed")


def analyze_placement(
    golden: np.ndarray,
    inspected: np.ndarray,
    reference_boxes: list[Box],
    candidate_boxes: list[Box],
) -> dict:
    """Run placement verification. See module docstring for the contract."""
    if golden.shape != (IMAGE_SIZE, IMAGE_SIZE) or \
            inspected.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise PlacementInputError("both images must be 512x512")
    n_refs, n_cands = len(reference_boxes), len(candidate_boxes)

    # ---- 1. Registration + exposure trust (shared pipeline) ---------------
    base = inspect_images(golden, inspected, include_debug=False)
    if base["status"] == "undeterminable":
        return _whole_board_refusal(reference_boxes, candidate_boxes, base)

    warp = np.array(base["correction"]["warp_matrix"], dtype=np.float64)
    fit = ExposureFit(
        gain=float(base["correction"]["gain"]),
        bias=float(base["correction"]["bias"]),
        robust_sigma=float(base["correction"]["robust_sigma"]),
        inlier_ratio=float(base["metrics"]["exposure"]["inlier_ratio"]),
        iterations=int(base["metrics"]["exposure"]["irls_iterations"]),
    )

    # Inspected board rendered into golden-frame geometry and intensity scale.
    inspected_g = geometry.render_inspected_in_golden(inspected, warp)
    corrected_g = correct_to_reference(inspected_g, fit)
    inv_warp = geometry.invert_warp(warp)
    coverage_g, _ = geometry.overlap_masks(warp)

    # Candidate centers mapped back into golden-frame coordinates.
    cand_centers_g: list[tuple[float, float]] = []
    for box in candidate_boxes:
        cx, cy = box.center
        gx, gy = inv_warp @ np.array([cx, cy, 1.0])
        cand_centers_g.append((float(gx), float(gy)))

    # ---- 2. Geometric one-to-one assignment -------------------------------
    distance = np.zeros((n_refs, n_cands), dtype=np.float64)
    for i, ref in enumerate(reference_boxes):
        rcx, rcy = ref.center
        for j, (gx, gy) in enumerate(cand_centers_g):
            distance[i, j] = float(np.hypot(rcx - gx, rcy - gy))
    assignment, leftovers = _geometric_assignment(distance, n_refs, n_cands)

    # ---- 3. Texture scoring in canonical golden-frame windows -------------
    # Every comparison uses the *reference box's own index-aligned* 32x32
    # window on the back-rendered, exposure-corrected inspected board. This
    # avoids rounding the mapped candidate center to a shifted pixel grid,
    # which measurably degrades both NCC and MAE on sharp outlines.
    golden_patches = [
        golden[b.y:b.y + b.h, b.x:b.x + b.w] for b in reference_boxes
    ]
    content_patches = [
        corrected_g[b.y:b.y + b.h, b.x:b.x + b.w] for b in reference_boxes
    ]
    # Row i: observed content at reference i; column k: golden component k.
    # The diagonal therefore holds the in-place identity scores.
    ncc_mat = np.zeros((n_refs, n_refs), dtype=np.float64)
    mae_mat = np.zeros((n_refs, n_refs), dtype=np.float64)
    for i in range(n_refs):
        for k in range(n_refs):
            ncc_mat[i, k] = _ncc(golden_patches[k], content_patches[i])
            mae_mat[i, k] = _mae(golden_patches[k], content_patches[i])

    # ---- 4. Per-reference verdicts ----------------------------------------
    results: list[dict] = []
    for i, ref in enumerate(reference_boxes):
        entry: dict = {
            "reference_index": i,
            "reference_box": ref.as_list(),
            "candidate_box": None,
            "candidate_index": None,
            "verdict": None,
            "reason_code": None,
            "reason": None,
            "scores": None,
        }
        # Windows partly outside the warp overlap contain border-fill pixels
        # rather than real content; such a position cannot be judged.
        overlap_ratio = float(
            (coverage_g[ref.y:ref.y + ref.h, ref.x:ref.x + ref.w] > 0).mean()
        )
        if overlap_ratio < MIN_WINDOW_OVERLAP:
            entry.update({
                "verdict": "undeterminable",
                "reason_code": "INSUFFICIENT_WINDOW_OVERLAP",
                "reason": (
                    f"only {overlap_ratio:.0%} of the reference window lies "
                    "inside the trusted registration overlap; the component "
                    "cannot be judged"
                ),
                "scores": {"window_overlap_ratio": round(overlap_ratio, 3)},
            })
            if i in assignment:
                j = assignment[i]
                entry["candidate_box"] = candidate_boxes[j].as_list()
                entry["candidate_index"] = j
            results.append(entry)
            continue
        if i not in assignment:
            # No candidate covers this footprint. Missing only when the
            # corrected content at the golden window clearly differs from the
            # golden component; otherwise refuse (a shifted/unlisted part may
            # still be present).
            ncc_self = float(ncc_mat[i, i])
            mae_self = float(mae_mat[i, i])
            nearest = float(distance[i].min()) if n_cands else None
            entry["scores"] = {
                "ncc": round(ncc_self, 4), "mae": round(mae_self, 3),
                "nearest_candidate_distance_px": (
                    round(nearest, 2) if nearest is not None else None),
            }
            if ncc_self <= MISSING_NCC_MAX and mae_self >= MISSING_MAE_MIN:
                entry["verdict"] = "missing"
                entry["reason_code"] = "NO_CANDIDATE_AT_POSITION"
                entry["reason"] = (
                    "no candidate box covers this reference position and the "
                    "corrected footprint content does not match the golden "
                    "component"
                )
            else:
                entry["verdict"] = "undeterminable"
                entry["reason_code"] = "NO_GEOMETRIC_CANDIDATE"
                entry["reason"] = (
                    "no candidate box is close enough to this reference "
                    "position, but the footprint still resembles the golden "
                    "component; cannot declare it missing"
                )
            results.append(entry)
            continue

        j = assignment[i]
        own = (float(ncc_mat[i, i]), float(mae_mat[i, i]))
        cross = [
            (float(ncc_mat[i, k]), float(mae_mat[i, k]))
            for k in range(n_refs) if k != i
        ]
        runner_ncc = max((s[0] for s in cross), default=-1.0)
        nearest_mae = min((s[1] for s in cross), default=np.inf)
        verdict, reason = _identity_verdict(
            golden_patches[i], own, cross
        )
        if verdict == "correct":
            reason_code = None
        elif verdict == "wrong":
            reason_code = "WRONG_COMPONENT"
        else:
            reason_code = reason.split(":", 1)[0]
        entry.update({
            "candidate_box": candidate_boxes[j].as_list(),
            "candidate_index": j,
            "verdict": verdict,
            "reason_code": reason_code,
            "reason": reason,
            "scores": {
                "ncc": round(own[0], 4),
                "mae": round(own[1], 3),
                "ncc_margin_to_runner_up": round(own[0] - runner_ncc, 4),
                "mae_margin_to_runner_up": round(nearest_mae - own[1], 3),
                "center_distance_px": round(float(distance[i, j]), 3),
            },
        })
        results.append(entry)

    metrics = {
        # Rows = observed content at reference i, cols = golden component k.
        "identity_ncc_matrix": np.round(ncc_mat, 4).tolist(),
        "identity_mae_matrix": np.round(mae_mat, 3).tolist(),
        "center_distance_matrix": np.round(distance, 3).tolist(),
        "ecc_correlation": float(base["correction"]["ecc_correlation"]),
        "overlap_ratio": float(base["correction"]["overlap_ratio"]),
        "gain": fit.gain,
        "bias": fit.bias,
        "unassigned_candidate_indices": leftovers,
    }
    return {
        "status": "ok",
        "reason_code": None,
        "reason": None,
        "results": results,
        "unassigned_candidates": [
            {"candidate_index": j, "candidate_box": candidate_boxes[j].as_list()}
            for j in leftovers
        ],
        "metrics": metrics,
    }


def _whole_board_refusal(
    reference_boxes: list[Box], candidate_boxes: list[Box], base: dict
) -> dict:
    """Every position is refused when registration/exposure is untrusted."""
    results = []
    for i, ref in enumerate(reference_boxes):
        results.append({
            "reference_index": i,
            "reference_box": ref.as_list(),
            "candidate_box": None,
            "candidate_index": None,
            "verdict": "undeterminable",
            "reason_code": base["reason_code"],
            "reason": (
                "whole-board refusal: "
                + (base["reason"] or "registration/exposure not trustworthy")
            ),
            "scores": None,
        })
    return {
        "status": "undeterminable",
        "reason_code": base["reason_code"],
        "reason": base["reason"],
        "results": results,
        "unassigned_candidates": [
            {"candidate_index": j, "candidate_box": box.as_list()}
            for j, box in enumerate(candidate_boxes)
        ],
        "metrics": {"base_inspection": base["metrics"]},
    }
