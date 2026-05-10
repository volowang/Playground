# CS526 Interactivity Report

## Summary

- Dataset path: `/work2/09796/jingxinw7692/frontera/data_analytics/Playground/rlva/outputs/course_dataset/rlva_trace_corpus_close_to_1gb.csv`.
- Probe case: `inventory` / `dqn` seed `11` compared against `ppo`.
- Backend weighted throughput: 433,333 records per second.
- Backend peak throughput: 439,024 records per second.
- Front-end median response: 86.8 ms.
- Front-end slowest response: 873.3 ms.
- Sub-second front-end actions: 4/4.
- One-second interaction target satisfied: `yes`.

## Measurement Breakdown

- `Scan raw course dataset` [backend]: 4203600 records in 9.5749 s (439,024 records/s). Sequential CSV scan of the stored course dataset.
- `Load benchmark report table` [backend]: 25 records in 0.0143 s (1,752 records/s). Reads the aggregated evaluation table used by the dashboard.
- `Load seeded summary` [backend]: 12 records in 0.0040 s (3,025 records/s). Loads one precomputed window-summary artifact for live inspection.
- `Compute policy comparison clusters` [backend]: 24 records in 0.1077 s (223 records/s). Clusters two controller summaries to prepare coordinated comparison views.
- `Build behavior-space view` [frontend]: 12 records in 0.8733 s (14 records/s). Constructs the linked scatter plot used for focused anomaly inspection.
- `Build temporal-trend view` [frontend]: 12 records in 0.0696 s (173 records/s). Constructs the linked timeline view for the focused summary windows.
- `Build benchmark-overview view` [frontend]: 25 records in 0.1040 s (240 records/s). Constructs the multi-environment overview used in the validation narrative.
- `Build policy-comparison view` [frontend]: 8 records in 0.0596 s (134 records/s). Constructs the cluster-level comparison chart for the active probe case.

