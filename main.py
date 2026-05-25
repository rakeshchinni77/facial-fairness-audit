"""Entrypoint for the facial-fairness-audit project.

TODO: connect orchestration logic in later phases.
"""

from __future__ import annotations

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


logger = logging.getLogger(__name__)


def main() -> None:
    """Log a placeholder startup message for the foundation scaffold."""

    logger.info("facial-fairness-audit foundation scaffold is ready.")


if __name__ == "__main__":
    main()