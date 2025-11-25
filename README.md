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

Interactive Streamlit application for antibody sequencing workflows. The Sequencing tab currently supports AbNatiV nativeness scoring of heavy-chain sequences and exposes placeholders for future Nanokink integration.

## 1. Prerequisites

| Requirement                     | Notes                                                                                                                                                    |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| macOS / Linux with Python 3.12+ | Project is tested on macOS Sonoma + Python 3.13 via venv.                                                                                                |
| Requirements file               | `pip install -r requirements.txt` installs Streamlit, AbNatiV, pdbfixer, etc. Install ANARCI separately via `python scripts/install_vendored_anarci.py`. |
| Xcode CLT / build tools         | Needed to compile portions of ANARCI on Apple Silicon.                                                                                                   |
| HMMER 3.3+, wget, curl          | Used by the ANARCI build pipeline to download germlines/HMMs. Install via `brew install hmmer wget`.                                                     |
| Conda or pip                    | We use a `python -m venv` virtualenv, but a conda env works too.                                                                                         |

### Optional (GPU / structure workflows)

- OpenMM + pdbfixer (via `pip install git+https://github.com/pandegroup/openmm` and `pdbfixer`).
- AB/NanoBuilder toolchain if you plan to run humanisation commands from the AbNatiV submodule.

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

Install everything (Streamlit UI, AbNatiV, pdbfixer, etc.) from the provided requirements file:

```bash
pip install -r requirements.txt
```

### Install the vendored ANARCI wheel

We removed ANARCI from `requirements.txt` to keep Hugging Face builds deterministic. Install the vendored wheel locally via the helper script:

```bash
python scripts/install_vendored_anarci.py
```

Re-run the script with `--force` if you need to reinstall the package into a fresh virtual environment.

## 4. Initialize AbNatiV cache

The PyPI `abnativ` package ships the CLI and Python helpers we use in `services/abnativ_client.py`. After installing requirements, download the pretrained checkpoints once per user:

```bash
abnativ init  # stores weights in ~/.abnativ by default
```

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

   - Commit the provided `requirements.txt` (installs Streamlit, AbNatiV, pdbfixer, etc.) and the `postBuild` script (which decodes and installs the vendored ANARCI wheel).
   - Keep `external/` empty in Git—the Space will install AbNatiV/ANARCI from pip instead of pulling vendored sources.
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
