---
title: Sormanni Sequencing
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
---

# Sequencing UI

# Sequencing UI

Interactive Streamlit application for antibody sequencing workflows. The Sequencing tab exposes AbNatiV nativeness scoring for VH/VL sequences and NanoKink kink-probability predictions for VHH nanobodies, including CSV uploads and batch processing.

## 1. Prerequisites

| Requirement                       | Notes                                                                                                                                                          |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| macOS / Linux with Python 3.12+   | Project is tested on macOS Sonoma + Python 3.13 via venv.                                                                                                      |
| Requirements file                 | `pip install -r requirements.txt` installs the Streamlit UI + core dependencies. Install ANARCI, AbNatiV, and NanoKink via `scripts/install_external_deps.py`. |
| Vendored externals                | The `external/` folder includes ready-to-use copies of ANARCI, AbNatiV, and NanoKink. No submodule sync is required beyond a normal `git clone`.               |
| Xcode CLT / build tools           | Needed to compile portions of ANARCI on Apple Silicon.                                                                                                         |
| HMMER 3.3+, MUSCLE 5+, wget, curl | Used by the ANARCI build pipeline to download germlines/HMMs. Install via `brew install hmmer brewsci/bio/muscle wget`.                                        |
| Conda or pip                      | We use a `python -m venv` virtualenv, but a conda env works too.                                                                                               |

### Optional (GPU / structure workflows)

- OpenMM + pdbfixer (via `pip install git+https://github.com/pandegroup/openmm` and `pdbfixer`).
- AB/NanoBuilder toolchain if you plan to run humanisation commands from the upstream AbNatiV toolkit.

## 2. Clone repository

```bash
git clone git@gitlab.developers.cam.ac.uk:group/sequencing.git
cd sequencing
```

## 3. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Install dependencies

Install everything (Streamlit UI, pdbfixer, etc.) from the provided requirements file:

```bash
pip install -r requirements.txt
```

### Install external ANARCI / AbNatiV / NanoKink

The `external/` directory already contains the vendored sources. Install all three packages from source using the helper script:

```bash
python scripts/install_external_deps.py
```

### Populate ANARCI germlines / HMMs

The upstream ANARCI repository does **not** commit IMGT germlines, so the editable install from `external/ANARCI` will fail with `ModuleNotFoundError: anarci.germlines` until you build the assets locally. Run the pipeline once on every fresh checkout:

```bash
# assumes the virtualenv lives in .venv; update PATH if you use conda
REPO=/path/to/sequence
cd "$REPO"/external/ANARCI/build_pipeline
PATH="$REPO/.venv/bin:$PATH" bash RUN_pipeline.sh

# copy the generated files into the package that pip installs in editable mode
cd "$REPO"  # repo root
cp external/ANARCI/build_pipeline/curated_alignments/germlines.py external/ANARCI/lib/python/anarci/
mkdir -p external/ANARCI/lib/python/anarci/dat
cp -R external/ANARCI/build_pipeline/HMMs external/ANARCI/lib/python/anarci/dat/
```

The script depends on MUSCLE, HMMER, wget, and curl; on macOS install them via Homebrew (`brew install brewsci/bio/muscle hmmer wget curl`). Once the files exist under `external/ANARCI/lib/python/anarci/`, re-run `python scripts/install_external_deps.py` to refresh the editable install inside your environment.

````

## 4. Initialize AbNatiV cache

The `external/AbNatiV` checkout provides the CLI and Python helpers we use in `services/abnativ_client.py`. After installing requirements + external dependencies (including NanoKink), download the pretrained checkpoints once per user:

```bash
abnativ init  # stores weights in ~/.abnativ by default
````

> **Note for Apple Silicon**: the published `anarci` wheel now bundles its HMM assets. If you still encounter missing `ALL.hmm` errors, reinstall ANARCI (`pip install --force-reinstall anarci`) or follow the upstream instructions to rebuild HMMs with `hmmer`.

## 5. Run the Streamlit app

```bash
./.venv/bin/streamlit run app.py
```

The `.streamlit/config.toml` already enables file watching, so saving Python files reloads the UI. Navigate to the _Sequencing_ page, enter a VH sequence (AHo alignment will run via ANARCI), and click **Run** to fetch nativeness scores from the local AbNatiV weights.

## 6. Troubleshooting

| Symptom                                           | Fix                                                                                                              |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `ALL.hmm` missing / `anarci` fails                | Rerun the ANARCI build pipeline and copy `HMMs` + `germlines.py` into your site-packages as above.               |
| `openmm` / `pdbfixer` import errors               | Install from conda-forge or pip (Apple Silicon requires `pip install git+https://github.com/pandegroup/openmm`). |
| ANARCI discards sequence (`header needed change`) | The sequence contains large insertions; trim or align manually—AbNatiV expects ≤149 AHo positions.               |
| Streamlit duplicate key errors                    | Clear browser cache or restart Streamlit after UI modifications.                                                 |

## 7. Contact form email notifications

The **Contact Us** page sends a notification email via SMTP when a visitor submits the form. Add your SMTP credentials to `.streamlit/secrets.toml` so Streamlit can load them at runtime:

```toml
[smtp]
host = "smtp.gmail.com"
port = 587
username = "your-gmail@example.com"
password = "app-specific-password"
sender = "your-gmail@example.com"
sender_name = "Sequence Contact"
recipient = "hussainjr.ali@gmail.com"
use_tls = true
```

- `use_ssl` can be set to `true` if your provider requires SMTPS (set `use_tls` to `false` in that case).
- Generate an app password when using Gmail or any provider that enforces OAuth/MFA.
- The `recipient` defaults to `hussainjr.ali@gmail.com`, but you can override it to forward messages elsewhere.
- Alternative (for managed hosts such as Hugging Face Spaces): set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_SENDER`, `SMTP_SENDER_NAME`, `SMTP_RECIPIENT`, `SMTP_USE_TLS`, and/or `SMTP_USE_SSL` as environment variables instead of committing a secrets file.

## 8. Deploying on Hugging Face Spaces

1. **Prepare the repo**

   - Commit the provided `requirements.txt` (installs Streamlit, pdbfixer, etc.) and the `postBuild` script (which installs ANARCI/AbNatiV/NanoKink from the vendored `external/` sources and runs `abnativ init`).
   - Verify `abnativ init` succeeds locally so you know the package download works on your platform.

2. **Create the Space**

   - In the Hugging Face UI click **New Space**, pick the **Streamlit** template, choose **CPU** hardware, and give it a name (e.g., `sormanni-sequencing`).
   - Copy the git URL shown on the Space page (`https://huggingface.co/spaces/<org>/sormanni-sequencing`).

3. **Push code to the Space**

   - Authenticate once with `huggingface-cli login`.
   - Add the Space as a remote: `git remote add hf https://huggingface.co/spaces/<org>/sormanni-sequencing`.
   - Push your branch: `git push hf main`. Spaces will automatically run `pip install -r requirements.txt` and launch `streamlit run app.py`.

4. **Configure SMTP secrets in the Space**

   - Open **Settings → Repository secrets** on the Space and add the SMTP\_\* variables listed above. Streamlit will read them at runtime via `st.secrets` fallback.
   - The `postBuild` script already runs `abnativ init` for both `user` and `root`, but if the cache is still empty you can manually run `abnativ init` from the Space’s **Logs → Shell** tab.

5. **Optional custom domain**
   - Paid Spaces tiers allow attaching a custom domain; follow HF instructions to add the DNS CNAME once the app is stable.
