from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRECTORIES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "artifacts",
    PROJECT_ROOT / "artifacts" / "plots",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "submission",
    PROJECT_ROOT / "configs",
]


logger = logging.getLogger(__name__)


def verify_required_directories() -> None:
    missing = [str(path) for path in REQUIRED_DIRECTORIES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required project directories: {missing}")


def verify_runtime_imports() -> None:
    import cv2  # noqa: F401
    import fairlearn  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401


def verify_project_startup() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import main

    main.verify_basic_imports()


def main_entry() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger.info("Starting Docker smoke validation")
    verify_required_directories()
    verify_runtime_imports()
    verify_project_startup()
    logger.info("Docker smoke validation completed successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main_entry())
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        logger.exception("Docker smoke validation failed: %s", exc)
        raise SystemExit(1) from exc
