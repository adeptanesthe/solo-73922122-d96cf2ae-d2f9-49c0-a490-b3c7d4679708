"""Fixed-seed synthetic placement-verification scenarios.

These helpers build 512x512 grayscale boards carrying up to six visually
distinct 32x32 SMD-like components at known reference positions, then
produce an inspected board with a controlled assembly outcome:

* correct placement (optionally with affine warp + exposure drift);
* a two-component swap (both positions must be reported ``wrong``);
* a three-way rotation of components;
* a missing component (footprint repainted with background);
* duplicated / look-alike components (identity must be refused, not guessed).

Like :mod:`tests.synth.generator`, this module lives strictly on the test
side of the boundary: the detector never imports it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 512
BOX = 32
BASE_SEED = 20261118

# Six spread-out, mutually non-overlapping 32x32 footprints.
REFERENCE_POSITIONS: list[tuple[int, int]] = [
    (80, 80), (250, 80), (420, 80),
    (80, 260), (250, 260), (420, 260),
]

COMPONENT_KINDS = ("bars", "grid", "diag", "blob", "rings", "cross")


def _rng(index: int) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + index)


def _background(rng: np.random.Generator) -> np.ndarray:
    img = np.full((SIZE, SIZE), 90.0, np.float32)
    small = rng.normal(0.0, 1.0, (16, 16)).astype(np.float32)
    img += cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC) * 12.0
    for _ in range(28):
        x0 = int(rng.integers(20, SIZE - 80))
        y0 = int(rng.integers(20, SIZE - 80))
        length = int(rng.integers(40, 200))
        thick = int(rng.integers(3, 9))
        gray = float(rng.integers(120, 210))
        if rng.random() < 0.5:
            img[y0:y0 + thick, x0:min(SIZE, x0 + length)] = gray
        else:
            img[y0:min(SIZE, y0 + length), x0:x0 + thick] = gray
    for _ in range(46):
        cx = int(rng.integers(20, SIZE - 20))
        cy = int(rng.integers(20, SIZE - 20))
        cv2.circle(img, (cx, cy), int(rng.integers(4, 12)),
                   float(rng.integers(30, 235)), thickness=-1)
    return img


def make_component(rng: np.random.Generator, kind: str) -> np.ndarray:
    """One deterministic, visually distinct 32x32 component texture."""
    p = np.full((BOX, BOX), float(rng.integers(60, 200)), np.float32)
    if kind == "bars":
        for k in range(0, BOX, 4):
            p[:, k:k + 2] = float(rng.integers(20, 240))
    elif kind == "grid":
        p[::5, :] = rng.integers(20, 80)
        p[:, ::5] = rng.integers(180, 240)
    elif kind == "diag":
        xx, yy = np.meshgrid(np.arange(BOX), np.arange(BOX))
        p = (((xx + yy) % 9 < 4) * 160 + 50).astype(np.float32)
    elif kind == "blob":
        cv2.circle(p, (16, 16), 11, 230, -1)
        cv2.circle(p, (11, 12), 4, 30, -1)
    elif kind == "rings":
        cv2.circle(p, (16, 16), 13, 40, 2)
        cv2.circle(p, (16, 16), 8, 220, 2)
        cv2.circle(p, (16, 16), 3, 60, -1)
    elif kind == "cross":
        p[13:19, :] = rng.integers(20, 70)
        p[:, 13:19] = rng.integers(190, 240)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown component kind {kind!r}")
    p += rng.normal(0.0, 2.0, (BOX, BOX))
    return np.clip(p, 0.0, 255.0).astype(np.uint8)


@dataclass
class PlacementScenario:
    name: str
    golden: np.ndarray
    inspected: np.ndarray
    reference_boxes: list[list[int]]      # [x, y, 32, 32] on golden
    candidate_boxes: list[list[int]]      # [x, y, 32, 32] on inspected
    expected: dict[int, str]              # reference index -> expected verdict
    true_warp: np.ndarray
    gain: float
    bias: float


def _warp_forward(img: np.ndarray, warp: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        img, warp.astype(np.float32), (SIZE, SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _random_warp(rng: np.random.Generator) -> tuple[np.ndarray, float, float, float]:
    angle = float(rng.uniform(-2.5, 2.5))
    tx = float(rng.uniform(-6.0, 6.0))
    ty = float(rng.uniform(-6.0, 6.0))
    warp = cv2.getRotationMatrix2D(((SIZE - 1) / 2.0,) * 2, angle, 1.0)
    warp[0, 2] += tx
    warp[1, 2] += ty
    return warp.astype(np.float64), angle, tx, ty


def _build_golden(
    rng: np.random.Generator, n_components: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    golden = _background(rng)
    components = []
    for k in range(n_components):
        comp = make_component(rng, COMPONENT_KINDS[k])
        components.append(comp)
        x, y = REFERENCE_POSITIONS[k]
        golden[y:y + BOX, x:x + BOX] = comp
    golden = np.clip(
        golden + rng.normal(0.0, 3.0, (SIZE, SIZE)), 0, 255
    ).astype(np.uint8)
    return golden, components


def _candidate_boxes(
    warp: np.ndarray, positions: list[tuple[int, int]]
) -> list[list[int]]:
    boxes = []
    for x, y in positions:
        mx, my = warp @ np.array([x + BOX / 2.0, y + BOX / 2.0, 1.0])
        boxes.append([int(round(mx - BOX / 2.0)),
                      int(round(my - BOX / 2.0)), BOX, BOX])
    return boxes


def _finalize(
    name: str,
    golden: np.ndarray,
    assembled: np.ndarray,
    warp: np.ndarray,
    rng: np.random.Generator,
    candidate_positions: list[tuple[int, int]],
    expected: dict[int, str],
    gain: float,
    bias: float,
) -> PlacementScenario:
    warped = _warp_forward(assembled, warp)
    inspected = gain * warped.astype(np.float32) + bias
    inspected += rng.normal(0.0, 2.0, (SIZE, SIZE)).astype(np.float32)
    inspected = np.clip(inspected, 0.0, 255.0).astype(np.uint8)
    ref_boxes = [[x, y, BOX, BOX] for (x, y) in
                 REFERENCE_POSITIONS[:len(expected)]]
    return PlacementScenario(
        name=name,
        golden=golden,
        inspected=inspected,
        reference_boxes=ref_boxes,
        candidate_boxes=_candidate_boxes(warp, candidate_positions),
        expected=expected,
        true_warp=warp,
        gain=gain,
        bias=bias,
    )


def make_correct_scenario(index: int = 0, n_components: int = 4) -> PlacementScenario:
    rng = _rng(index)
    golden, _ = _build_golden(rng, n_components)
    warp, _, _, _ = _random_warp(rng)
    gain, bias = float(rng.uniform(0.85, 1.15)), float(rng.uniform(-12.0, 12.0))
    expected = {k: "correct" for k in range(n_components)}
    return _finalize(
        "placement_correct", golden, golden.copy(), warp, rng,
        REFERENCE_POSITIONS[:n_components], expected, gain, bias,
    )


def make_swap_scenario(index: int = 100) -> PlacementScenario:
    """Two components exchange positions; both must be reported wrong."""
    rng = _rng(index)
    n = 4
    golden, components = _build_golden(rng, n)
    assembled = golden.copy()
    x0, y0 = REFERENCE_POSITIONS[0]
    x1, y1 = REFERENCE_POSITIONS[1]
    assembled[y0:y0 + BOX, x0:x0 + BOX] = components[1]
    assembled[y1:y1 + BOX, x1:x1 + BOX] = components[0]
    warp, _, _, _ = _random_warp(rng)
    gain, bias = 1.1, -8.0
    expected = {0: "wrong", 1: "wrong", 2: "correct", 3: "correct"}
    return _finalize(
        "placement_swap", golden, assembled, warp, rng,
        REFERENCE_POSITIONS[:n], expected, gain, bias,
    )


def make_three_way_scenario(index: int = 200) -> PlacementScenario:
    """Components rotate 0->1, 1->2, 2->0; position 3 stays correct."""
    rng = _rng(index)
    n = 4
    golden, components = _build_golden(rng, n)
    assembled = golden.copy()
    for ref_idx, comp_idx in ((0, 1), (1, 2), (2, 0)):
        x, y = REFERENCE_POSITIONS[ref_idx]
        assembled[y:y + BOX, x:x + BOX] = components[comp_idx]
    warp, _, _, _ = _random_warp(rng)
    gain, bias = 0.92, 7.0
    expected = {0: "wrong", 1: "wrong", 2: "wrong", 3: "correct"}
    return _finalize(
        "placement_three_way", golden, assembled, warp, rng,
        REFERENCE_POSITIONS[:n], expected, gain, bias,
    )


def make_missing_scenario(index: int = 300) -> PlacementScenario:
    """Component at position 2 is absent; no candidate box covers it."""
    rng = _rng(index)
    n = 4
    golden, _ = _build_golden(rng, n)
    assembled = golden.copy()
    x, y = REFERENCE_POSITIONS[2]
    plain_bg = _background(np.random.default_rng(BASE_SEED + 777))
    assembled[y:y + BOX, x:x + BOX] = plain_bg[y:y + BOX, x:x + BOX]
    warp, _, _, _ = _random_warp(rng)
    gain, bias = 1.05, -4.0
    candidate_positions = [REFERENCE_POSITIONS[k] for k in (0, 1, 3)]
    expected = {0: "correct", 1: "correct", 2: "missing", 3: "correct"}
    return _finalize(
        "placement_missing", golden, assembled, warp, rng,
        candidate_positions, expected, gain, bias,
    )


def make_duplicate_scenario(index: int = 400) -> PlacementScenario:
    """Positions 0 and 1 carry byte-identical components.

    Their identities cannot be told apart, so only those two positions are
    refused; the distinct components at positions 2/3 remain decidable.
    """
    rng = _rng(index)
    n = 4
    golden, components = _build_golden(rng, n)
    x1, y1 = REFERENCE_POSITIONS[1]
    golden[y1:y1 + BOX, x1:x1 + BOX] = components[0].copy()
    warp, _, _, _ = _random_warp(rng)
    gain, bias = 0.97, 5.0
    expected = {0: "undeterminable", 1: "undeterminable",
                2: "correct", 3: "correct"}
    return _finalize(
        "placement_duplicate", golden, golden.copy(), warp, rng,
        REFERENCE_POSITIONS[:n], expected, gain, bias,
    )


def make_lookalike_scenario(index: int = 600) -> PlacementScenario:
    """Two components are deliberately near-identical look-alikes.

    The identity margin between them is small, so only those two positions
    are refused; the remaining distinct components stay fully decidable.
    """
    rng = _rng(index)
    n = 4
    golden, components = _build_golden(rng, n)
    # Derive a look-alike of component 0: the same structure with only a
    # localized intensity perturbation, close enough that the identity
    # margin between the two golden components stays below the refusal
    # threshold (measured NCC margin ~0.01, threshold 0.10).
    lookalike = components[0].copy()
    band = lookalike[24:26, :].astype(np.int16) + 25
    lookalike[24:26, :] = np.clip(band, 0, 255).astype(np.uint8)
    x1, y1 = REFERENCE_POSITIONS[1]
    golden[y1:y1 + BOX, x1:x1 + BOX] = lookalike
    warp, _, _, _ = _random_warp(rng)
    gain, bias = 1.05, -6.0
    expected = {0: "undeterminable", 1: "undeterminable",
                2: "correct", 3: "correct"}
    return _finalize(
        "placement_lookalike", golden, golden.copy(), warp, rng,
        REFERENCE_POSITIONS[:n], expected, gain, bias,
    )


def make_flat_parts_scenario(index: int = 700) -> PlacementScenario:
    """Two flat, low-texture components (structure channel uninformative).

    The MAE / intensity-level channel must still separate a correct flat
    part (mean level ~120) from a swapped flat part (mean level ~200);
    swapping the two flats must produce two ``wrong`` verdicts while the
    textured components at the other positions stay correct.
    """
    rng = _rng(index)
    n = 4
    golden, components = _build_golden(rng, n)
    flat_a = np.clip(
        np.full((BOX, BOX), 120.0) + rng.normal(0.0, 1.5, (BOX, BOX)),
        0, 255
    ).astype(np.uint8)
    flat_b = np.clip(
        np.full((BOX, BOX), 200.0) + rng.normal(0.0, 1.5, (BOX, BOX)),
        0, 255
    ).astype(np.uint8)
    # Replace the first two textured components with the flat pair.
    for comp, ref_idx in ((flat_a, 0), (flat_b, 1)):
        x, y = REFERENCE_POSITIONS[ref_idx]
        golden[y:y + BOX, x:x + BOX] = comp
    golden = np.clip(
        golden + rng.normal(0.0, 3.0, (SIZE, SIZE)), 0, 255
    ).astype(np.uint8)

    # Swap the two flat parts on the assembled board.
    assembled = golden.copy()
    x0, y0 = REFERENCE_POSITIONS[0]
    x1, y1 = REFERENCE_POSITIONS[1]
    assembled[y0:y0 + BOX, x0:x0 + BOX] = flat_b
    assembled[y1:y1 + BOX, x1:x1 + BOX] = flat_a

    warp, _, _, _ = _random_warp(rng)
    gain, bias = 1.1, -8.0
    expected = {0: "wrong", 1: "wrong", 2: "correct", 3: "correct"}
    return _finalize(
        "placement_flat_swap", golden, assembled, warp, rng,
        REFERENCE_POSITIONS[:n], expected, gain, bias,
    )


def make_solid_board_scenario(index: int = 500) -> PlacementScenario:
    """Solid-color board: registration is untrusted -> whole-board refusal."""
    rng = _rng(index)
    n = 3
    level = int(rng.integers(90, 170))
    golden = np.full((SIZE, SIZE), level, np.uint8)
    inspected = np.clip(
        golden.astype(np.float32) * 1.05 + 4.0, 0, 255
    ).astype(np.uint8)
    warp = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -2.0]])
    expected = {k: "undeterminable" for k in range(n)}
    return PlacementScenario(
        name="placement_solid_refusal",
        golden=golden,
        inspected=inspected,
        reference_boxes=[[x, y, BOX, BOX] for (x, y) in
                         REFERENCE_POSITIONS[:n]],
        candidate_boxes=_candidate_boxes(
            warp, REFERENCE_POSITIONS[:n]),
        expected=expected,
        true_warp=warp,
        gain=1.05,
        bias=4.0,
    )
