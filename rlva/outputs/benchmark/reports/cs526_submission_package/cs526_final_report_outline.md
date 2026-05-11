# CS526 Final Report Outline

## 1. Project Identity

- Product name: `Decision System Review Studio`.
- Analytic engine: `RLVA: Behavior-Level Diagnostics for Sequential Decision Policies`.
- Course framing: interactive visual analytics system plus reproducible benchmark pipeline.
- Target users: traffic analysts, inventory planners, service managers, control QA engineers, robotics safety reviewers, supervisors, and technical reviewers.

## 2. Data And Scale

- Raw course dataset target: close to 1 GB.
- Processed benchmark trace records: 75000.
- Processed benchmark summary windows: 1500.
- Describe why long sequential-decision traces require summarization before interactive exploration.

## 3. Fundamental Data Questions

- Which system windows look behaviorally risky?
- When does the selected system change operating regime?
- Which controllers behave differently even when reward differences are small?

## 4. System Design

- Back-end: trace collection, window summarization, anomaly scoring, shift localization, controller comparison, ablation, robustness evaluation, and task-performance aggregation.
- Front-end: coordinated views where every main chart is clickable, linking ranked incident evidence, behavior space, temporal trends, comparison plots, notebook annotations, task logging, and export panels.
- Execution scenario: a user selects one of five systems, inspects risky windows, compares controllers, and exports decision-ready evidence.

## 5. Evaluation

- Main anomaly result: traffic / ppo with anomaly AUC 0.937 +/- 0.021.
- Main shift result: traffic / pg with shift AUC 0.940 +/- 0.020.
- Strongest policy-separation result: traffic heuristic vs pg with top-cluster JS divergence 0.688 +/- 0.011.
- Compare RLVA against reward-only or simpler external baselines.

## 6. Interactivity And Usability

- Backend weighted throughput: 433,333 records per second.
- Front-end median response: 86.8 ms.
- Front-end slowest response: 873.3 ms.
- Completed user tasks logged: 3.
- Median task completion time: 58.0 seconds.
- Support-signal rate: 100.0%.
- Explain that click selections update focused windows, signals, systems/controllers, score metrics, and comparison groups while staying within the classroom-response target.

## 7. Limitations And Next Steps

- Explain that RLVA is a behavior-analysis engine, not a claim of state-of-the-art RL reward performance.
- Propose broader traffic datasets, more policy families, larger user studies, or streaming extensions as follow-up work.

## 8. Reproducibility Appendix

- Environment activation: `source .venv/bin/activate`.
- Demo command: `PYTHONPATH=. streamlit run rlva/src/app.py`.
- Export command: `PYTHONPATH=. python -m rlva.src.run_paper_pipeline --steps evaluate,aggregate,ablation,robustness,exports`.

