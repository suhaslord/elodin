"""One-off diagnostics for the Voyager 1 Jupiter validation review.

This file lives on an analysis branch, not the review PR. It is intentionally
broader than the focused PR so we can isolate whether the observed Chapter 1 / 2
ranking depends on epoch, SPICE load order, or how the moving planetary sources
are sampled inside each RK4 step.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import spiceypy as spice

G = 6.6743e-11
SUN_MASS = 1.9885e30
STEP_SECONDS = 3600.0
SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"
ENCOUNTER_KERNEL = SPICE_DIR / "vgr1_jup230.bsp"
DE440_KERNEL = SPICE_DIR / "de440.bsp"
LSK_KERNEL = SPICE_DIR / "naif0012.tls"
EXPECTED_ENCOUNTER_SHA256 = "e1ea3f72f19b15508bc45979771a36a97d02f33056b76867d444304cb82205c9"
FRAME = "ECLIPJ2000"
OBSERVER = "SUN"
PROBE = "VOYAGER 1"
PROBE_ID = -31

PLANETS = (
    ("MERCURY BARYCENTER", 3.3011e23),
    ("VENUS BARYCENTER", 4.8675e24),
    ("EARTH", 5.97219e24),
    ("MARS BARYCENTER", 6.4171e23),
    ("JUPITER BARYCENTER", 1.898125e27),
    ("SATURN BARYCENTER", 5.6834e26),
    ("URANUS BARYCENTER", 8.6813e25),
    ("NEPTUNE BARYCENTER", 1.02413e26),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_kernels(order: str) -> list[str]:
    spice.kclear()
    spice.furnsh(str(LSK_KERNEL))
    if order == "encounter_then_de440":
        sequence = [ENCOUNTER_KERNEL, DE440_KERNEL]
    elif order == "de440_then_encounter":
        sequence = [DE440_KERNEL, ENCOUNTER_KERNEL]
    else:
        raise ValueError(order)
    for kernel in sequence:
        spice.furnsh(str(kernel))
    return [path.name for path in sequence]


def coverage_utc() -> list[list[str]]:
    window = spice.spkcov(str(ENCOUNTER_KERNEL), PROBE_ID)
    intervals = []
    for index in range(spice.wncard(window)):
        left, right = spice.wnfetd(window, index)
        intervals.append(
            [
                spice.et2utc(left, "ISOC", 3),
                spice.et2utc(right, "ISOC", 3),
            ]
        )
    return intervals


def direct_acceleration(position_m: np.ndarray, source_m: np.ndarray, mu: float) -> np.ndarray:
    delta = source_m - position_m
    distance = np.linalg.norm(delta)
    return mu * delta / distance**3


def chapter_acceleration(chapter: int, position_m: np.ndarray, source_positions_m: tuple[np.ndarray, ...]) -> np.ndarray:
    total = direct_acceleration(position_m, np.zeros(3), G * SUN_MASS)
    for (_, mass), source_m in zip(PLANETS, source_positions_m, strict=True):
        mu = G * mass
        direct = direct_acceleration(position_m, source_m, mu)
        if chapter == 1:
            total += direct
        elif chapter == 2:
            source_distance = np.linalg.norm(source_m)
            total += direct - mu * source_m / source_distance**3
        else:
            raise ValueError(chapter)
    return total


def source_state(epoch_et: float) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    states = []
    for name, _ in PLANETS:
        state_km, _ = spice.spkezr(name, epoch_et, FRAME, "NONE", OBSERVER)
        states.append(
            (
                np.asarray(state_km[:3], dtype=np.float64) * 1000.0,
                np.asarray(state_km[3:], dtype=np.float64) * 1000.0,
            )
        )
    return tuple(states)


def truth_state(epoch_et: float) -> np.ndarray:
    state_km, _ = spice.spkezr(PROBE, epoch_et, FRAME, "NONE", OBSERVER)
    state = np.asarray(state_km, dtype=np.float64)
    return np.concatenate((state[:3] * 1000.0, state[3:] * 1000.0))


def rk4_step_exact_spice(chapter: int, epoch_et: float, state: np.ndarray) -> np.ndarray:
    half = STEP_SECONDS / 2.0

    def derivative(candidate: np.ndarray, sample_et: float) -> np.ndarray:
        positions = tuple(position for position, _ in source_state(sample_et))
        return np.concatenate((candidate[3:], chapter_acceleration(chapter, candidate[:3], positions)))

    k1 = derivative(state, epoch_et)
    k2 = derivative(state + half * k1, epoch_et + half)
    k3 = derivative(state + half * k2, epoch_et + half)
    k4 = derivative(state + STEP_SECONDS * k3, epoch_et + STEP_SECONDS)
    return state + STEP_SECONDS * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def rk4_step_elodin_semantics(chapter: int, epoch_et: float, state: np.ndarray) -> np.ndarray:
    """Match the Voyager example's pre_step + six_dof source-body behavior.

    Planet state is refreshed from SPICE once at the start of the tick; during
    the RK4 stages the ephemeris body then drifts with that sampled velocity.
    """
    sampled = source_state(epoch_et)
    half = STEP_SECONDS / 2.0

    def positions(offset: float) -> tuple[np.ndarray, ...]:
        return tuple(position + velocity * offset for position, velocity in sampled)

    def derivative(candidate: np.ndarray, offset: float) -> np.ndarray:
        return np.concatenate(
            (candidate[3:], chapter_acceleration(chapter, candidate[:3], positions(offset)))
        )

    k1 = derivative(state, 0.0)
    k2 = derivative(state + half * k1, half)
    k3 = derivative(state + half * k2, half)
    k4 = derivative(state + STEP_SECONDS * k3, STEP_SECONDS)
    return state + STEP_SECONDS * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def residual(state: np.ndarray, truth: np.ndarray) -> dict:
    position_m = state[:3] - truth[:3]
    velocity_mps = state[3:] - truth[3:]
    return {
        "position_error_km": float(np.linalg.norm(position_m) / 1000.0),
        "velocity_error_mps": float(np.linalg.norm(velocity_mps)),
        "position_residual_km": [float(value / 1000.0) for value in position_m],
        "velocity_residual_mps": [float(value) for value in velocity_mps],
    }


def run_case(start_utc: str, elapsed_days: tuple[int, ...], source_mode: str) -> dict:
    start_et = spice.utc2et(start_utc)
    initial = truth_state(start_et)
    states = {1: initial.copy(), 2: initial.copy()}
    checkpoints = {day * 86400 for day in elapsed_days}
    max_elapsed = max(checkpoints)
    records = {"1": [], "2": []}

    for elapsed in range(0, max_elapsed + int(STEP_SECONDS), int(STEP_SECONDS)):
        if elapsed in checkpoints:
            truth = truth_state(start_et + elapsed)
            for chapter in (1, 2):
                records[str(chapter)].append(
                    {
                        "elapsed_days": elapsed / 86400.0,
                        "utc": spice.et2utc(start_et + elapsed, "ISOC", 3),
                        **residual(states[chapter], truth),
                    }
                )
        if elapsed == max_elapsed:
            break
        epoch_et = start_et + elapsed
        for chapter in (1, 2):
            if source_mode == "exact_spice_substeps":
                states[chapter] = rk4_step_exact_spice(chapter, epoch_et, states[chapter])
            elif source_mode == "elodin_pre_step":
                states[chapter] = rk4_step_elodin_semantics(chapter, epoch_et, states[chapter])
            else:
                raise ValueError(source_mode)

    comparisons = []
    for ch1, ch2 in zip(records["1"], records["2"], strict=True):
        ch1_error = ch1["position_error_km"]
        ch2_error = ch2["position_error_km"]
        comparisons.append(
            {
                "elapsed_days": ch1["elapsed_days"],
                "chapter2_minus_chapter1_km": ch2_error - ch1_error,
                "chapter2_vs_chapter1_pct": 0.0 if ch1_error == 0.0 else 100.0 * (ch2_error - ch1_error) / ch1_error,
            }
        )
    return {
        "start_utc": start_utc,
        "source_mode": source_mode,
        "chapters": records,
        "comparisons": comparisons,
    }


def sweep(source_mode: str) -> list[dict]:
    starts = (
        "1979-02-06T00:00:00",
        "1979-02-10T00:00:00",
        "1979-02-14T00:00:00",
        "1979-02-18T00:00:00",
        "1979-02-21T00:00:00",
        "1979-02-24T00:00:00",
        "1979-02-28T00:00:00",
        "1979-03-02T00:00:00",
    )
    rows = []
    for start in starts:
        case = run_case(start, (0, 1, 2, 3), source_mode)
        final1 = case["chapters"]["1"][-1]
        final2 = case["chapters"]["2"][-1]
        rows.append(
            {
                "start_utc": start,
                "end_utc": final1["utc"],
                "chapter1_position_error_km": final1["position_error_km"],
                "chapter2_position_error_km": final2["position_error_km"],
                "chapter2_minus_chapter1_km": final2["position_error_km"] - final1["position_error_km"],
                "chapter2_vs_chapter1_pct": 100.0 * (final2["position_error_km"] - final1["position_error_km"]) / final1["position_error_km"],
            }
        )
    return rows


def run_all() -> dict:
    missing = [str(path) for path in (LSK_KERNEL, ENCOUNTER_KERNEL, DE440_KERNEL) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing kernels: " + ", ".join(missing))

    encounter_hash = sha256(ENCOUNTER_KERNEL)
    result = {
        "encounter_sha256": encounter_hash,
        "encounter_sha256_matches": encounter_hash == EXPECTED_ENCOUNTER_SHA256,
        "orders": {},
    }

    for order in ("encounter_then_de440", "de440_then_encounter"):
        loaded = load_kernels(order)
        order_result = {
            "loaded_spk_order": loaded,
            "voyager1_coverage_utc": coverage_utc(),
            "feb06_exact": run_case("1979-02-06T00:00:00", (0, 1, 2), "exact_spice_substeps"),
            "feb06_elodin_semantics": run_case("1979-02-06T00:00:00", (0, 1, 2), "elodin_pre_step"),
            "feb21_exact": run_case("1979-02-21T00:00:00", (0, 3, 7, 11, 12, 13), "exact_spice_substeps"),
            "feb21_elodin_semantics": run_case("1979-02-21T00:00:00", (0, 3, 7, 11, 12, 13), "elodin_pre_step"),
            "three_day_sweep_exact": sweep("exact_spice_substeps"),
            "three_day_sweep_elodin_semantics": sweep("elodin_pre_step"),
        }
        result["orders"][order] = order_result

    spice.kclear()
    return result


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, sort_keys=True))
