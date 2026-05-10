# RLVA Large Course Dataset

- Dataset path: `rlva/outputs/course_dataset/rlva_trace_corpus_close_to_1gb.csv.gz`
- Delivery format: split gzip archive parts committed to GitHub and reassembled locally at runtime
- Actual size: `1020000997` bytes
- Total records: `4203600`
- Bytes per record: `242.649`
- Source trace files: `125`
- Replica passes over benchmark traces: `57`
- Data source: Synthetic large raw trace corpus generated from seeded RLVA benchmark traces with provenance columns preserved.

## Environment Inventory

- `cartpole` source traces: 25
- `inventory` source traces: 25
- `lunarlander` source traces: 25
- `queue` source traces: 25
- `traffic` source traces: 25

## Course Requirement Check

- CS526 dataset-size target: close to `1 GB`.
- This generated corpus satisfies that target with `1020000997` bytes.
