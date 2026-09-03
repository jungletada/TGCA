"""Unit tests for canonical direct-statistics verification."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.lazy_assignment.verify_experiment1_canonical import direct_statistics


def test_direct_statistics_match_numpy_reference() -> None:
    values = np.linspace(-0.75, 0.9, 784, dtype=np.float32)
    result = direct_statistics(values)

    assert result["score_mean"] == pytest.approx(np.mean(values, dtype=np.float64))
    assert result["score_std"] == pytest.approx(np.std(values.astype(np.float64)))
    assert result["score_max"] == pytest.approx(0.9)
    assert result["score_q50"] == pytest.approx(np.quantile(values, 0.50))
    assert result["score_q95"] == pytest.approx(np.quantile(values, 0.95))

