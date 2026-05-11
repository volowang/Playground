# CS526 Final Project Brief

## Project Identity

- Product name: `Decision System Review Studio`.
- Analytic engine: `RLVA: Behavior-Level Diagnostics for Sequential Decision Policies`.
- Project type: user-facing visual analytics system plus reproducible experiment pipeline.
- Primary target users: operations analysts, inventory planners, service managers, control QA engineers, robotics safety reviewers, and supervisors who review controller behavior.
- Core execution scenario: a user selects one of five decision systems, clicks visual evidence to focus a window, controller, signal, or behavior group, compares the active controller against an alternate controller, and exports a focused decision memo.

## Fundamental Data Questions

- Which time windows are risky enough to require user review?
- When does the selected system appear to change operating regime?
- Which controller is safer or more stable for the selected system?

## Dataset and Processing Scale

- Environments represented in benchmark outputs: cartpole (25 seeds), inventory (25 seeds), lunarlander (25 seeds), queue (25 seeds), traffic (25 seeds).
- Policy families represented in benchmark outputs: dqn (25 runs), heuristic (25 runs), pg (25 runs), ppo (25 runs), random (25 runs).
- Benchmark trace files: 125.
- Benchmark summary files: 125.
- Total trace records processed: 75000.
- Total summary windows processed: 1500.
- Trace storage footprint: 15127460 bytes.

- Course-scale raw dataset path: `rlva/outputs/course_dataset/rlva_trace_corpus_close_to_1gb.csv.gz`.
- Course-scale raw dataset records: 4203600.
- Course-scale raw dataset size: 1020000997 bytes.
- Course-scale raw dataset bytes per record: 242.649.
- Course-scale data source: Synthetic large raw trace corpus generated from seeded RLVA benchmark traces with provenance columns preserved.
## Interface and Interaction Design

- Front-end: Streamlit dashboard focused on five user-facing scenarios; course evidence is exported as a static report instead of being exposed inside the product UI.
- Coordinated views: system selector, risk-change matrix, priority queue, action mix, timeline, behavior-space plot, focused-signal chart, benchmark heatmap, controller scorecard, controller-comparison charts, case notes, and review exports.
- Interaction components: every visible Plotly chart supports click selection; clicks update the focused window, action filter, selected signal, selected system/controller, score metric, or comparison group.
- Interactivity target: precomputed benchmark artifacts are loaded from CSV so classroom interactions stay in the sub-second to few-second range instead of re-running training online.

## Development Stack

- Back-end tools: Python, pandas, scikit-learn, PyTorch, Gymnasium, Plotly export utilities, and reproducible benchmark scripts under `rlva/src/`.
- Front-end tools: Streamlit plus Plotly for interactive charts.
- Reproducibility assets: seeded benchmark traces, summary CSVs, benchmark tables, ablation tables, robustness tables, and exported PNG/PDF figures.

## Measured Interactivity

- Backend weighted throughput: 433,333 records per second.
- Backend peak throughput: 439,024 records per second.
- Front-end median response: 86.8 ms.
- Front-end slowest measured response: 873.3 ms.
- Sub-second actions: 4/4.
- One-second classroom interaction target satisfied: `yes`.

## User Task Performance

- Completed user tasks logged: 3.
- Median task completion time: 58.0 seconds.
- Median focus changes per task: 3.0.
- Median filter changes per task: 2.0.
- Average operator confidence: 4.33/5.
- Support-signal rate: 100.0%.
- Export usage rate: 100.0%.

## Quantitative Evaluation Highlights

- Best anomaly-detection result: `traffic` / `ppo` with anomaly AUC 0.937 +/- 0.021.
- Best shift-detection result: `traffic` / `pg` with shift AUC 0.940 +/- 0.020.
- Best shift-localization result: `traffic` / `pg` with localization error 0.200 +/- 0.400.
- Largest gain over the reward-collapse baseline: `lunarlander` / `heuristic` with anomaly-AUC improvement 0.491.
- Strongest external anomaly baseline: `reward_jump` on `inventory` / `heuristic` with anomaly AUC 0.975 +/- 0.050.
- Strongest policy-separation case: `traffic` `heuristic` vs `pg` with top-cluster JS divergence 0.688 +/- 0.011 at K=8.
- Robustness example: `lunarlander` / `pg` at window length 75 reaches shift hit@3 1.000 +/- 0.000.

## Ablation Takeaways

- `cartpole` / `dqn`: best feature configuration is state_action (anomaly AUC 0.646 +/- 0.108).
- `cartpole` / `heuristic`: best feature configuration is state_action (anomaly AUC 0.566 +/- 0.108).
- `cartpole` / `pg`: best feature configuration is state_only (anomaly AUC 0.606 +/- 0.137).
- `cartpole` / `ppo`: best feature configuration is state_only (anomaly AUC 0.543 +/- 0.139).
- `cartpole` / `random`: best feature configuration is no_switch_rate (anomaly AUC 0.606 +/- 0.111).
- `inventory` / `dqn`: best feature configuration is state_action (anomaly AUC 0.977 +/- 0.033).
- `inventory` / `heuristic`: best feature configuration is full (anomaly AUC 0.914 +/- 0.026).
- `inventory` / `pg`: best feature configuration is state_only (anomaly AUC 0.880 +/- 0.075).
- `inventory` / `ppo`: best feature configuration is reward_only (anomaly AUC 0.897 +/- 0.082).
- `inventory` / `random`: best feature configuration is no_switch_rate (anomaly AUC 0.903 +/- 0.062).
- `lunarlander` / `dqn`: best feature configuration is no_switch_rate (anomaly AUC 0.783 +/- 0.039).
- `lunarlander` / `heuristic`: best feature configuration is full (anomaly AUC 0.709 +/- 0.100).

## Final Demo Checklist

- Activate the virtual environment in this repository: `source .venv/bin/activate`.
- Run the dashboard: `PYTHONPATH=. streamlit run rlva/src/app.py`.
- Regenerate the paper and course artifacts when needed: `PYTHONPATH=. python -m rlva.src.run_paper_pipeline --steps evaluate,aggregate,ablation,robustness,exports`.
- Export this course brief again after new experiments: `PYTHONPATH=. python -m rlva.src.export_course_deliverables`.

## Why This Scores Well In CS526 Terms

- Value of extracted information: the system helps concrete users across five decision systems isolate risky windows, confirm behavior shifts, compare controllers, and leave with a documented action instead of only a chart.
- Methods and models: the project combines RL policies, intervention-aware trace collection, window-level summarization, clustering, anomaly scoring, shift localization, ablation, baseline comparison, and explicit task-performance logging.
- Interactivity: every main chart is clickable and linked, so the dashboard behaves like a coordinated visual analysis tool rather than a static report.
