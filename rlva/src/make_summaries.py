from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rlva.src.behavior import summarize_trace
from rlva.src.config import ENVIRONMENTS, POLICIES, SUMMARY_DIR, TRACE_DIR, resolve_summary_path, resolve_trace_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build windowed behavior summaries for one environment")
    parser.add_argument("--env", type=str, required=True, choices=ENVIRONMENTS)
    parser.add_argument("--L", type=int, default=50)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--out-dir", type=Path, default=SUMMARY_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for policy in POLICIES:
        trace_path = resolve_trace_path(env_name=args.env, policy=policy, trace_dir=args.trace_dir)
        if not trace_path.exists():
            raise FileNotFoundError("Missing trace file: {0}".format(trace_path))
        summary_df = summarize_trace(trace=pd.read_csv(trace_path), L=args.L)
        out_path = resolve_summary_path(env_name=args.env, policy=policy, summary_dir=args.out_dir)
        summary_df.to_csv(out_path, index=False)
        print("Saved: {0} ({1} rows)".format(out_path, len(summary_df)))


if __name__ == "__main__":
    main()
