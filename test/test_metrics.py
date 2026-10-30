"""Validation suite for src/lab/evaluation/metrics.py.

Run directly:

    uv run python test/test_metrics.py

Each check uses a case whose answer is known by hand: a circular orbit for k,
and small lists of losses for the decision metrics.
"""

import numpy as np

from lab.data.gcrs_data_generation import two_body
from lab.evaluation.metrics import MU, best_threshold, rates_at, true_k

# A circular equatorial orbit with a 24 h period.
PERIOD_S = 86400.0
RADIUS_KM = (MU * (PERIOD_S / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
R1 = np.array([RADIUS_KM, 0.0, 0.0])
V1 = np.array([0.0, np.sqrt(MU / RADIUS_KM), 0.0])


def k_after(hours):
    r2, _ = two_body(MU, hours * 3600.0, R1, V1)
    return true_k(R1, V1, r2, hours * 3600.0)


def test_true_k_short_way():
    assert k_after(6.0) == 0.0, "90 deg is k = 0"
    assert k_after(11.0) == 0.0, "165 deg is k = 0"


def test_true_k_long_way():
    assert k_after(13.0) == 0.5, "195 deg is k = 0.5"
    assert k_after(23.0) == 0.5, "345 deg is k = 0.5"


def test_true_k_counts_revolutions():
    assert k_after(30.0) == 1.0, "450 deg is k = 1"
    assert k_after(42.0) == 1.5, "630 deg is k = 1.5"
    assert k_after(54.0) == 2.0, "810 deg is k = 2"
    assert k_after(66.0) == 2.5, "990 deg is k = 2.5"


def test_rates_at_threshold():
    rates = rates_at(np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 5.0]), 3.0)
    assert rates["false_negative_percent"] == 25.0, "one of four true pairs rejected"
    assert rates["false_positive_percent"] == 50.0, "one of two false pairs accepted"


def test_best_threshold_separates_when_possible():
    threshold = best_threshold(np.array([1.0, 2.0]), np.array([5.0, 6.0]))
    assert threshold == 2.0, f"threshold {threshold}"


def test_best_threshold_minimises_errors_when_pairs_overlap():
    threshold = best_threshold(np.array([1.0, 2.0, 100.0]), np.array([5.0, 6.0]))
    assert threshold == 2.0, f"threshold {threshold}"


TESTS = [
    test_true_k_short_way,
    test_true_k_long_way,
    test_true_k_counts_revolutions,
    test_rates_at_threshold,
    test_best_threshold_separates_when_possible,
    test_best_threshold_minimises_errors_when_pairs_overlap,
]


def main():
    passed = 0
    failed = []

    for test in TESTS:
        name = test.__name__
        try:
            test()
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}\n          {exc}")
        except Exception as exc:  # noqa: BLE001 -- report and continue
            failed.append(name)
            print(f"  ERROR {name}\n          {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  pass  {name}")

    print()
    print(f"{passed}/{len(TESTS)} passed")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
