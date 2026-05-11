# CS526 Presentation Storyboard

## Slide 1. User And Problem

- Introduce the five primary user groups: traffic, inventory, queueing, control QA, and robotics safety reviewers.
- State the product name and the three user questions: risky window, behavior-change boundary, and safer controller.

## Slide 2. Data

- Show that the project uses a persistent course-scale raw trace corpus close to 1 GB and explain why time-varying traces need summarization.
- Mention 75000 processed trace records and 125 benchmark trace files.

## Slide 3. Workflow Demo

- Show the user flow: choose system -> click a chart -> inspect focused evidence -> compare controllers -> export memo.
- Emphasize that the interface is coordinated, task-driven, and produces work products for an operator.

## Slide 4. Live Incident Triage Story

- Lead with one selected system, then connect it to the strongest anomaly evidence: traffic / ppo anomaly AUC 0.937 +/- 0.021.
- Use ranked evidence, the behavior map, and the timeline to connect the number back to a specific risky window.

## Slide 5. Behavior Shift Confirmation

- Lead with traffic / pg shift AUC 0.940 +/- 0.020.
- Show temporal evidence and explain where the operating-regime boundary appears.

## Slide 6. Why This Is More Than A Dashboard

- Compare against reward_jump on inventory / heuristic (0.975 +/- 0.050).
- Explain why behavior summaries plus operator workflow expose structure that reward-only views miss.

## Slide 7. Usability And Interactivity

- Report completed user tasks: 3.
- Report median task completion time: 58.0 seconds.
- Report backend weighted throughput: 433,333 records/s.
- Report front-end median response: 86.8 ms.
- State that every main Plotly chart is clickable and linked to a user decision.
- State clearly that operators can both complete tasks and interact within the one-second classroom-response target by reusing precomputed summaries.

## Slide 8. Closing

- Summarize the value of extracted information, methods/models used, task performance, and interface interactivity.
- End with the five-system user dashboard inside the Streamlit application.

