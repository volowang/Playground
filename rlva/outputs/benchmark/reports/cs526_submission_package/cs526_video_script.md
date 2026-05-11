# CS526 Video Script

## 0:00-0:30 Opening

This project is Decision System Review Studio, a user-facing visual analytics system powered by RLVA. Instead of trusting only total reward, the system helps users inspect risky windows, detect operating changes, and compare controllers behaviorally across five decision systems.

## 0:30-1:10 Data And Method

The project uses a persistent raw trace corpus close to one gigabyte, plus seeded benchmark summaries for five environments and five policy families. RLVA converts long traces into window-level summaries, then uses those windows for incident ranking, shift localization, and cross-strategy comparison.

## 1:10-2:00 Evidence

One strong anomaly example is traffic / ppo with anomaly AUC 0.937 +/- 0.021. One strong shift-detection example is traffic / pg with shift AUC 0.940 +/- 0.020. These results are backed by clickable linked views that move from benchmark scores to concrete windows, selected controllers, signals, and comparison groups.

## 2:00-2:40 Usability And Interactivity

For course evaluation, the system also measures whether users complete the intended tasks. The exported reports summarize 3 completed tasks, median task time around 58.0 seconds, backend throughput around 433,333 records per second, and median front-end response around 86.8 milliseconds.

## 2:40-3:20 Closing

The key contribution is not a higher reward score. The contribution is a five-system review product that extracts behavior-level evidence, supports interactive exploration, records task performance, and provides quantitative validation plus presentation-ready artifacts for classroom demonstration.

