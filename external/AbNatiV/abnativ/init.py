"""
 Copyright 2023. Aubin Ramon, Oliver Wissett and Pietro Sormanni. CC BY-NC-SA 4.0
"""

import argparse
import os
import subprocess
from typing import List


ZENODO_RECORD = "17295347"  # ← your actual Zenodo record ID
BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD}/files"


# Check what OS we are on
def get_platform():
    return os.uname().sysname

if get_platform() == "Linux" or get_platform() == "Darwin":
    # NOTE: Maybe use /usr/local/share/abnativ/models instead?
    PRETRAINED_MODELS_DIR = os.path.expanduser("~/.abnativ/models/pretrained_models")
elif get_platform() == "Windows":
    PRETRAINED_MODELS_DIR = os.path.expanduser(
        "~\\AppData\\Local\\abnativ\\models\\pretrained_models"
    )
else:
    raise Exception("Unsupported OS")



def ensure_zenodo_models(filenames: List[str], do_force_update : bool, 
                         model_dir: str = PRETRAINED_MODELS_DIR):
    '''If do_force_update will force redowloading every file.'''

    os.makedirs(model_dir, exist_ok=True)

    if do_force_update: 
        missing = filenames
        print(f"Forcing update: downloading all the {len(missing)} files from Zenodo…")
    
    else:
        missing = []
        for fname in filenames:
            local_path = os.path.join(model_dir, fname)
            if not os.path.isfile(local_path):
                missing.append(fname)
            else:
                print(f"✔ {fname} already present")

        if not missing:
            print("All model files are present; nothing to download.")
            return

        print(f"Downloading {len(missing)} missing files from Zenodo…")

    for fname in missing:
        url = f"{BASE_URL}/{fname}?download=1"
        target = os.path.join(model_dir, fname)
        cmd = [
            "wget",
            "--content-disposition",
            "--no-cache", 
            "-O", target,
            url
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
        print(f"✔ Downloaded {fname}")


def init(args: argparse.Namespace):
    expected = [
        "vhh2_model.ckpt",
        "vh2_model.ckpt",
        "vl2_model.ckpt",
        "vpaired2_model.ckpt",
        "vlambda_model.ckpt",
        "vkappa_model.ckpt",
        "vh_model.ckpt",
        "vhh_model.ckpt",
        "vh2_rhesus_model.ckpt",
    ]
    ensure_zenodo_models(expected, do_force_update=args.force_update)
    print(f"✅ PRETRAINED_MODELS_DIR set to: {PRETRAINED_MODELS_DIR}")