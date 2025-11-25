#!/usr/bin/env python3
"""Install ANARCI from the vendored base64 wheel.

This is a local counterpart to the Hugging Face `postBuild` logic. It decodes
`vendor/anarci-1.3.whl.b64` into a temporary wheel file and asks pip to install
it into the active Python environment.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENCODED_WHEEL = REPO_ROOT / "vendor" / "anarci-1.3.whl.b64"
WHEEL_FILENAME = "anarci-1.3-py3-none-any.whl"


def anarci_is_available() -> bool:
    """Return True when the `anarci` module is importable."""
    return importlib.util.find_spec("anarci") is not None


def decode_wheel(target_path: Path) -> None:
    """Decode the base64 wheel artifact to `target_path`."""
    if not ENCODED_WHEEL.exists():
        raise FileNotFoundError(
            f"Encoded wheel not found at {ENCODED_WHEEL}. Did you clone the vendor file?"
        )
    data = ENCODED_WHEEL.read_bytes()
    target_path.write_bytes(base64.b64decode(data))


def pip_install(wheel_path: Path, *, force: bool) -> None:
    """Run `pip install` on the decoded wheel."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
    ]
    if force:
        cmd.append("--force-reinstall")
    cmd.append(str(wheel_path))
    print(f"[installer] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even if ANARCI is already importable",
    )
    args = parser.parse_args()

    if anarci_is_available() and not args.force:
        print("[installer] ANARCI already available; use --force to reinstall.")
        return 0

    with tempfile.TemporaryDirectory(prefix="anarci-install-") as tmp_dir:
        wheel_path = Path(tmp_dir) / WHEEL_FILENAME
        print(f"[installer] Decoding wheel to {wheel_path}")
        decode_wheel(wheel_path)
        pip_install(wheel_path, force=args.force)

    print("[installer] ANARCI installation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
