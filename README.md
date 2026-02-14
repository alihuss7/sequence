# Sormanni Lab Sequencing Platform

> A modern web application for antibody sequence analysis powered by state-of-the-art machine learning models

## Overview

The **Sormanni Sequencing Platform** is an interactive Streamlit-based web interface that streamlines antibody sequencing workflows for researchers and scientists. This platform provides seamless access to specialized ML-powered endpoints—**AbNatiV**, **NbForge**, **NbFrame**, and **NanoMelt**—enabling rapid batch analysis of antibody sequences without requiring local model installations or GPU hardware.

### Key Features

- 🧬 **Multi-Model Analysis**: Unified interface for AbNatiV (nativeness scoring), NbForge (structure prediction), NbFrame (CDR3 conformation classification), and NanoMelt (thermal stability estimation)
- 📊 **Batch Processing**: Upload CSV files containing multiple sequences and process them efficiently
- 📈 **Interactive Results**: Sortable tables with detailed metrics and per-residue analysis
- 💾 **Data Export**: Download comprehensive results as CSV for further analysis
- 🔬 **Database Management**: Track and organize your sequencing runs
- 📧 **Contact Form**: Built-in support form with SMTP integration
- ☁️ **Cloud-Native**: Leverages managed API endpoints for scalable, maintenance-free deployment

Whether you're analyzing variable heavy (VH) chains, VHH nanobodies, or exploring antibody engineering candidates, this platform provides a fast, user-friendly interface to cutting-edge computational tools.

---

## Quick Start

```bash
git clone git@gitlab.developers.cam.ac.uk:group/sequencing.git
cd sequencing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# echo 'SEQUENCE_LIBRARIES_URL="https://<your-cloud-run-host>"' > .env
streamlit run app.py
```

AbNatiV, NbForge, NbFrame, and NanoMelt requests go through the lightweight `services/*_client.py` HTTP helpers and return results directly to the Sequencing page. The app requires `SEQUENCE_LIBRARIES_URL` so it knows which managed deployment to contact—grab the value from Cloud Run (or ask the platform team). Values placed in `.env` are loaded automatically via `python-dotenv`, so once you edit `.env` you no longer need to export anything manually.

---

## Managed API Endpoints

| Endpoint    | Description                                                                                   |
| ----------- | --------------------------------------------------------------------------------------------- |
| `/`         | Health probe returning basic service metadata.                                                |
| `/abnativ`  | Scores VH/VL/VHH sequences and returns nativeness values plus optional residue summaries.     |
| `/nbforge`  | Runs nanobody sequence-to-structure inference with optional minimization and NbFrame scoring. |
| `/nbframe`  | Classifies CDR3 conformation as kinked/extended/uncertain from sequence or structure inputs.  |
| `/nanomelt` | Estimates apparent melting temperatures for VHH nanobodies.                                   |

Each endpoint expects JSON containing the sequence(s) plus optional knobs such as `nativeness_type` or `do_alignment`. AbNatiV currently accepts one sequence per request, so the client submits each entry sequentially:

```bash
SERVICE_URL="https://<your-cloud-run-host>"
curl -X POST \
  "$SERVICE_URL/abnativ" \
  -H "Content-Type: application/json" \
  -d '{
        "sequence_id": "vh_example",
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISWNSGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKYPYYYGMDVWGQGTTVTVSS",
        "nativeness_type": "VH2",
        "do_align": true,
        "is_vhh": false
      }'
```

### Model Input Notes

| Model    | Best Used For                                                | Input Expectations                                                                     | Common Failure Mode                                             |
| -------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| AbNatiV  | Nativeness scoring of antibody variable domains              | Full variable-domain sequence (typically ~95-130 aa), standard amino-acid letters only | Short fragments can fail scoring/alignment on the backend       |
| NbForge  | Single-sequence nanobody structure prediction                | Full VHH sequence with antibody-like framework regions                                 | ANARCI numbering fails on non-antibody-like or truncated inputs |
| NbFrame  | CDR3 conformation classification (kinked/extended/uncertain) | Sequence suitable for antibody alignment/numbering                                     | Returns alignment error for malformed or short sequences        |
| NanoMelt | Apparent melting temperature estimate                        | Full nanobody sequence recommended                                                     | Inference can be slower and may hit timeout limits              |

### Environment Variables

| Variable                 | Default          | Purpose                                                               |
| ------------------------ | ---------------- | --------------------------------------------------------------------- |
| `SEQUENCE_LIBRARIES_URL` | unset (required) | Override when pointing at a different deployment or local dev server. |
| `.env`                   | not committed    | Create manually to store the variable above for reusable local runs.  |

Set these before launching Streamlit (or inside your hosting provider’s UI) to redirect traffic to staging/prod stacks.

---

## SMTP Settings (Contact Page)

The **Contact Us** form emails submissions via SMTP. Provide credentials either through `.streamlit/secrets.toml`:

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

…or environment variables (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_SENDER`, `SMTP_SENDER_NAME`, `SMTP_RECIPIENT`, `SMTP_USE_TLS`, `SMTP_USE_SSL`). Missing credentials trigger an inline warning but do not stop the main app.

---

## Deploying to Hugging Face Spaces (or similar)

1. Push this repository to your target Space (Streamlit template works best).
2. Ensure `postBuild` is executable; it simply runs `pip install -r requirements.txt`.
3. Configure `SEQUENCE_LIBRARIES_URL` under **Settings → Repository secrets** so the UI reaches the correct Cloud Run instance.
4. Add SMTP secrets if you want email notifications from the Contact page.

Because the heavy ML models now live behind the managed API, deployments are fast and do not require system packages, MUSCLE/HMMER, or GPU hardware.

---

## Troubleshooting

| Symptom                                                      | Suggested Fix                                                                                                                                |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Sequencing tab reports "processing failed for all sequences" | Check the Cloud Run logs, confirm the API URL is correct, and ensure the service allows your IP.                                             |
| Requests hang or time out                                    | The HTTP clients default to `(10s connect, 120s read)` timeouts. Reduce sequence batches or verify the remote service scaling configuration. |
| Contact form errors                                          | Provide SMTP credentials as described above or disable the button in `pages/contact_us.py`.                                                  |

Reach out via the Contact page or edit the `services/api_client.py` helper if you need to target a different API environment.
