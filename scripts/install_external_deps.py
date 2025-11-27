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
    clone_url: str | None = None
    editable: bool = True

    def install(self) -> None:
        """Install the dependency either from a local checkout or directly via git."""

        source, printable_source = self._resolve_source()
        print(f"[external-install] Installing {self.name} from {printable_source}")

        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
        if self.editable:
            cmd.append("-e")
        cmd.append(source)
        subprocess.check_call(cmd)

    def _resolve_source(self) -> tuple[str, str]:
        if self.path is not None:
            if not self.path.exists():
                self._bootstrap_local_checkout()
            relative = str(self.path.relative_to(REPO_ROOT))
            return str(self.path), relative

        if self.git_url is not None:
            return self.git_url, self.git_url

        raise ValueError(f"Dependency {self.name} missing both path and git URL")

    def _bootstrap_local_checkout(self) -> None:
        if self.clone_url is None:
            raise FileNotFoundError(
                f"Expected directory {self.path} for {self.name} was not found "
                "and no clone URL has been provided."
            )

        print(f"[external-install] Cloning {self.name} into {self.path}")
        assert self.path is not None  # for type checkers
        self.path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", self.clone_url, str(self.path)]
        )


DEPENDENCIES = (
    ExternalDependency(
        "ANARCI",
        path=REPO_ROOT / "external" / "ANARCI",
        clone_url="https://github.com/oxpig/ANARCI.git",
    ),
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
    parser.add_argument(
        "--skip-abnativ-init",
        action="store_true",
        help="Install dependencies only; defer 'abnativ init' to a later step",
    )
    args = parser.parse_args()

    for dep in DEPENDENCIES:
        try:
            dep.install()
        except FileNotFoundError as exc:
            if args.allow_missing:
                print(f"WARNING: {exc}")
                continue
            raise

    if args.skip_abnativ_init:
        print("[external-install] Skipping AbNatiV pretrained model init (per flag)")
    else:
        print(
            "[external-install] Initialising AbNatiV pretrained models via 'abnativ init'"
        )
        subprocess.check_call([sys.executable, "-m", "abnativ", "init"])

    print("[external-install] External dependency installation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
