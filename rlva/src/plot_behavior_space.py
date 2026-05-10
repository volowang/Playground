from __future__ import annotations

import argparse

from rlva.src.compare import load_summary
from rlva.src.config import ENVIRONMENTS, FIG_DIR, POLICIES
from rlva.src.viz import build_behavior_space_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Export a behavior-space figure using the shared visualization pipeline")
    parser.add_argument("--env", type=str, required=True, choices=ENVIRONMENTS)
    parser.add_argument("--policy", type=str, required=True, choices=POLICIES)
    parser.add_argument("--color-by", type=str, default="anomaly_score")
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = load_summary(env_name=args.env, policy=args.policy)
    figure = build_behavior_space_figure(summary_df, env_name=args.env, color_by=args.color_by)
    out_path = FIG_DIR / "{0}__{1}_behavior_space.html".format(args.env, args.policy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print("Saved: {0}".format(out_path))


if __name__ == "__main__":
    main()
