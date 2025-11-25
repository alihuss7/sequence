"""Setuptools configuration for the Sormanni Sequencing Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import List

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent


def _read_requirements() -> List[str]:
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        return []
    return [
        line.strip()
        for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _read_readme() -> str:
    readme_file = PROJECT_ROOT / "README.md"
    if not readme_file.exists():
        return ""
    return readme_file.read_text(encoding="utf-8")


setup(
    name="sequence-app",
    version="0.1.0",
    description="Streamlit interface for Sormanni sequencing workflows",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    url="https://huggingface.co/spaces/alihuss7/sequence",
    packages=find_packages(exclude=("external*", "tests", "tests.*")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=_read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: Streamlit",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
