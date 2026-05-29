from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def results_dir(project_root: Path) -> Path:
    return project_root / "results"


@pytest.fixture(scope="session")
def artifacts_dir(project_root: Path) -> Path:
    return project_root / "artifacts"


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    return project_root / "data"
