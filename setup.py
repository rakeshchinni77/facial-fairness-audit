"""Packaging placeholder for facial-fairness-audit.

TODO: refine packaging metadata once the implementation phase begins.
"""

from setuptools import find_packages, setup


setup(
    name="facial-fairness-audit",
    version="0.1.0",
    description="Foundation package for facial verification fairness auditing",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
)