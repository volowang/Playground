# CS526 Submission Package

This directory contains the course-facing materials that make the Decision System Review Studio final project easy to grade and easy to present.

## What Is Included

- `cs526_final_report_outline.md`: report structure aligned to the course rubric.
- `cs526_presentation_storyboard.md`: suggested slide-by-slide flow for the live demo.
- `cs526_video_script.md`: 3+ minute narration script for the required video.
- `cs526_submission_checklist.md`: final handoff checklist for code, data, report, and media.
- `course_task_performance_report.md`: exported evidence that operators can complete the intended review tasks.
- `../course_evidence_report.md`: static text-and-image replacement for the removed in-app course evidence workspace.

## Project Snapshot

- Benchmark traces: 125
- Benchmark summaries: 125
- Processed trace records: 75000
- Best anomaly story: traffic / ppo (0.937 +/- 0.021)
- Best shift story: traffic / pg (0.940 +/- 0.020)
- Completed user tasks logged: 3
- Median task time: 58.0 seconds
- Front-end median response: 86.8 ms
- Backend weighted throughput: 433,333 records per second

## Recommended Presentation Command

`source .venv/bin/activate && export PYTHONPATH=. && streamlit run rlva/src/app.py`

