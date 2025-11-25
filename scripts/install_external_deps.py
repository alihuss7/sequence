#!/usr/bin/env python3
"""Install ANARCI from the local ``external/ANARCI`` submodule.

This replaces the previous vendored-wheel flow by assuming the upstream ANARCI
repository is checked out as a git submodule under ``external/ANARCI``. The
helper shells out to pip with ``-e`` so edits to the submodule are reflected
immediately in the active environment.
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
    path: Path
    editable: bool = True

    def install(self) -> None:
        printable_path = self.path.relative_to(REPO_ROOT)
        print(f"[external-install] Installing {self.name} from {printable_path}")

        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
        if self.editable:
            cmd.append("-e")
        cmd.append(str(self.path))
        subprocess.check_call(cmd)


DEPENDENCIES = (ExternalDependency("ANARCI", REPO_ROOT / "external" / "ANARCI"),)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Print a warning instead of failing when a dependency folder is missing",
    )
    args = parser.parse_args()

    for dep in DEPENDENCIES:
        if not dep.path.exists():
            message = (
                f"[external-install] Expected directory {dep.path} was not found. "
                "Did you run 'git submodule update --init external/ANARCI'?"
            )
            if args.allow_missing:
                print(f"WARNING: {message}")
                continue
            raise FileNotFoundError(message)

        dep.install()

    print("[external-install] External dependency installation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
