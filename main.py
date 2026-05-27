"""Entrypoint for the facial-fairness-audit project.

TODO: connect orchestration logic in later phases.
"""

from __future__ import annotations

import logging
import sys
import time


def verify_basic_imports() -> None:
    """Verify the core runtime libraries are available at container startup."""

    import cv2  # noqa: F401
    import fairlearn  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


logger = logging.getLogger(__name__)


def main() -> None:
    """Log a placeholder startup message for the foundation scaffold."""

    try:
        verify_basic_imports()
    except Exception:  # pragma: no cover - startup guard for the container.
        logger.exception("Runtime dependency verification failed.")
        sys.exit(1)

    print("facial-fairness-audit foundation scaffold is ready.")
    logger.info("facial-fairness-audit foundation scaffold is ready.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("facial-fairness-audit received shutdown signal; exiting cleanly.")


if __name__ == "__main__":
    main()