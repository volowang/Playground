# Traffic Operations Review Studio

This repository contains the deployable dashboard version of `Traffic Operations Review Studio`, a traffic-operations visual analytics project built on top of `RLVA`.

The GitHub version includes:

- The dashboard source under `rlva/src/`
- The report assets and benchmark outputs required by the dashboard
- The full 1GB course dataset, stored in GitHub-safe split archive parts under `rlva/outputs/course_dataset/`

It does not include local virtual environments, caches, or unrelated course materials.

## Local Deployment

Use Python `3.10+`.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=.
streamlit run rlva/src/app.py
```

Open the Streamlit URL shown in the terminal, typically `http://localhost:8501`.

## Dataset Delivery

GitHub does not allow a normal Git blob larger than `100MB`, so the required 1GB dataset is committed as split archive parts:

- `rlva/outputs/course_dataset/rlva_trace_corpus_close_to_1gb.csv.gz.part-*`

The app automatically reassembles these parts into a local gzip archive on first use and reads the dataset from there. You do not need to manually unzip or rebuild the dataset before launching the dashboard.

## Project Layout

```text
rlva/
  src/                       dashboard and support code
  outputs/
    benchmark/               benchmark summaries, traces, figures, reports
    course_dataset/          1GB dataset archive parts and metadata
    figs/                    comparison HTML artifacts used by the dashboard
    report_assets/           supporting markdown/csv assets
    summaries/               precomputed summary tables
    tables/                  precomputed comparison tables
```

## Notes

- The repository is optimized for opening the dashboard locally.
- Training environments, model-building toolchains, and checked-in virtual environments are intentionally excluded from the GitHub deployment version.
