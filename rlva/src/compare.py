from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rlva.src.config import (
    BENCHMARK_POLICIES,
    BENCHMARK_SUMMARY_DIR,
    OUTPUTS_DIR,
    SUMMARY_FEATURE_COLUMNS,
    SUMMARY_DIR,
    TABLE_DIR,
    comparison_stem,
    resolve_summary_path,
)
from rlva.src.viz import build_comparison_figure


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-12, 1.0)
    q = np.clip(q.astype(float), 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two policies using cluster-level JS divergence")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--p1", type=str, required=True, choices=BENCHMARK_POLICIES)
    parser.add_argument("--p2", type=str, required=True, choices=BENCHMARK_POLICIES)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def load_summary(env_name: str, policy: str, summary_dir: Path = SUMMARY_DIR) -> pd.DataFrame:
    path = resolve_summary_path(env_name=env_name, policy=policy, summary_dir=summary_dir)
    if not path.exists():
        benchmark_dir = BENCHMARK_SUMMARY_DIR / env_name / policy
        benchmark_paths = sorted(benchmark_dir.glob("seed_*.csv"))
        if benchmark_paths:
            # Fall back to the first available benchmark seed summary when legacy single-file summaries are absent.
            return pd.read_csv(benchmark_paths[0])
        raise FileNotFoundError("Missing summary CSV for policy '{0}': {1}".format(policy, path))
    return pd.read_csv(path)


def compare_summary_clusters(df1: pd.DataFrame, df2: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    x_all = pd.concat([df1[SUMMARY_FEATURE_COLUMNS], df2[SUMMARY_FEATURE_COLUMNS]], ignore_index=True).astype(float)
    if x_all.empty:
        return pd.DataFrame()

    scaled = StandardScaler().fit_transform(x_all.to_numpy()) if len(x_all) >= 2 else np.zeros_like(x_all.to_numpy())

    def _cluster_once(cluster_count: int) -> pd.DataFrame:
        labels = KMeans(n_clusters=cluster_count, n_init=10, random_state=42).fit_predict(scaled)
        labels1 = labels[: len(df1)]
        labels2 = labels[len(df1) :]

        a = df1.copy()
        b = df2.copy()
        a["cluster"] = labels1
        b["cluster"] = labels2

        g1 = a.groupby("cluster", as_index=False).agg(
            n1=("cluster", "size"),
            q1=("q_bar", "mean"),
            mu1=("mu_bar", "mean"),
            sigma1=("sigma_bar", "mean"),
            rho1=("rho_bar", "mean"),
            p0_1=("freq0", "mean"),
            p1_1=("freq1", "mean"),
            p2_1=("freq2", "mean"),
            r1=("r_bar", "mean"),
            h1=("H_bar", "mean"),
            a1=("anomaly_score", "mean"),
        )
        g2 = b.groupby("cluster", as_index=False).agg(
            n2=("cluster", "size"),
            q2=("q_bar", "mean"),
            mu2=("mu_bar", "mean"),
            sigma2=("sigma_bar", "mean"),
            rho2=("rho_bar", "mean"),
            p0_2=("freq0", "mean"),
            p1_2=("freq1", "mean"),
            p2_2=("freq2", "mean"),
            r2=("r_bar", "mean"),
            h2=("H_bar", "mean"),
            a2=("anomaly_score", "mean"),
        )
        merged = g1.merge(g2, on="cluster", how="outer")
        merged["n1"] = merged["n1"].fillna(0)
        merged["n2"] = merged["n2"].fillna(0)
        both = merged["n1"].gt(0) & merged["n2"].gt(0)

        js_values: List[float] = []
        for _, row in merged.iterrows():
            if not (row["n1"] > 0 and row["n2"] > 0):
                js_values.append(np.nan)
                continue
            p = np.array([row["p0_1"], row["p1_1"], row["p2_1"]], dtype=float)
            q = np.array([row["p0_2"], row["p1_2"], row["p2_2"]], dtype=float)
            js_values.append(js_divergence(p, q))
        merged["js_div"] = js_values
        merged["n_total"] = merged["n1"] + merged["n2"]
        merged["reward_gap"] = (merged["r1"] - merged["r2"]).abs()
        merged["q_mean"] = 0.5 * (merged["q1"].fillna(0.0) + merged["q2"].fillna(0.0))
        merged["mu_mean"] = 0.5 * (merged["mu1"].fillna(0.0) + merged["mu2"].fillna(0.0))
        merged["sigma_mean"] = 0.5 * (merged["sigma1"].fillna(0.0) + merged["sigma2"].fillna(0.0))
        merged["rho_mean"] = 0.5 * (merged["rho1"].fillna(0.0) + merged["rho2"].fillna(0.0))
        merged["both_present"] = both
        merged["cluster_count"] = float(cluster_count)
        return merged.sort_values(["js_div", "reward_gap", "n_total"], ascending=[False, False, False], na_position="last").reset_index(drop=True)

    max_clusters = max(1, min(n_clusters, len(df1), len(df2), len(x_all)))
    fallback = None
    for cluster_count in range(max_clusters, 0, -1):
        result = _cluster_once(cluster_count)
        if fallback is None:
            fallback = result
        if result["js_div"].notna().any():
            return result
    return fallback if fallback is not None else pd.DataFrame()


def main() -> None:
    args = parse_args()
    if args.p1 == args.p2:
        raise ValueError("--p1 and --p2 must be different policies")

    df1 = load_summary(env_name=args.env, policy=args.p1, summary_dir=args.summary_dir)
    df2 = load_summary(env_name=args.env, policy=args.p2, summary_dir=args.summary_dir)
    table = compare_summary_clusters(df1, df2, n_clusters=args.K)

    tables_dir = args.out_dir / "tables"
    figs_dir = args.out_dir / "figs"
    stem = comparison_stem(env_name=args.env, p1=args.p1, p2=args.p2)
    csv_out = tables_dir / "{0}.csv".format(stem)
    html_out = figs_dir / "{0}.html".format(stem)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_out, index=False)
    build_comparison_figure(table, env_name=args.env, p1=args.p1, p2=args.p2).write_html(str(html_out), include_plotlyjs="cdn", full_html=True)

    top = table[table["js_div"].notna()].head(8)
    print("Saved table: {0} ({1} clusters)".format(csv_out, len(table)))
    print("Saved figure: {0}".format(html_out))
    print("\nTop clusters by JS divergence:")
    for _, row in top.iterrows():
        print(
            "  cluster={0:>2d} js={1:.4f} reward_gap={2:.4f} n={3:.0f}".format(
                int(row["cluster"]),
                float(row["js_div"]),
                float(row["reward_gap"]),
                float(row["n_total"]),
            )
        )


if __name__ == "__main__":
    main()
