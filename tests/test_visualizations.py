from __future__ import annotations

from tests.helpers import EXPECTED_PLOTS, PLOTS_DIR


def test_plots_directory_and_exports_exist():
    assert PLOTS_DIR.exists(), f"Plots directory missing: {PLOTS_DIR}"
    for name in EXPECTED_PLOTS:
        path = PLOTS_DIR / name
        assert path.exists(), f"Expected plot missing: {path}"
        assert path.stat().st_size > 0, f"Expected plot is empty: {path}"
