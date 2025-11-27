#!/usr/bin/env python3
"""Install ANARCI/AbNatiV/NanoKink/NanoMelt into the active environment.

ANARCI is installed from the vendored copy so the germline/HMM assets generated
at build time are available locally. The other tools are fetched directly from
their Git repositories via ``pip install git+https://...`` as requested.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExternalDependency:
    name: str
    path: Path | None = None
    git_url: str | None = None
    editable: bool = True

    def install(self) -> None:
        if self.git_url is None and self.path is None:
            raise ValueError(f"Dependency {self.name} missing both path and git URL")

        if self.git_url is not None:
            source = self.git_url
            printable_source = source
        else:
            assert self.path is not None  # for type checkers
            source = str(self.path)
            printable_source = str(self.path.relative_to(REPO_ROOT))

        print(f"[external-install] Installing {self.name} from {printable_source}")

        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
        if self.editable:
            cmd.append("-e")
        cmd.append(source)
        subprocess.check_call(cmd)


DEPENDENCIES = (
    ExternalDependency("ANARCI", path=REPO_ROOT / "external" / "ANARCI"),
    ExternalDependency(
        "AbNatiV",
        git_url="git+https://gitlab.developers.cam.ac.uk/ch/sormanni/abnativ.git#egg=abnativ",
        editable=False,
    ),
    ExternalDependency(
        "NanoKink",
        git_url="git+https://gitlab.developers.cam.ac.uk/ch/sormanni/nanomelt.git#egg=nanomelt",
        editable=False,
    ),
    ExternalDependency(
        "NanoMelt",
        git_url="git+https://gitlab.developers.cam.ac.uk/ch/sormanni/nanomelt.git#egg=nanomelt",
        editable=False,
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Print a warning instead of failing when a dependency folder is missing",
    )
    args = parser.parse_args()

    for dep in DEPENDENCIES:
        if dep.path is not None and not dep.path.exists():
            message = (
                f"[external-install] Expected directory {dep.path} was not found. "
                "Ensure the vendored external repositories exist inside 'external/'."
            )
            if args.allow_missing:
                print(f"WARNING: {message}")
                continue
            raise FileNotFoundError(message)

        dep.install()

    print("[external-install] Initialising AbNatiV pretrained models via 'abnativ init'")
    subprocess.check_call([sys.executable, "-m", "abnativ", "init"])

    print("[external-install] External dependency installation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
