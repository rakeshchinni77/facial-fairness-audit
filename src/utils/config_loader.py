"""Configuration loading utilities for future shared project settings.

The pipeline currently relies on explicit dataclass-based configuration in each
stage so command-line and notebook execution remain transparent. This module is
reserved for a future unified loader once the project adopts shared YAML-based
configuration files.
"""

from __future__ import annotations