from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from rlva.src.config import REPORT_ASSET_DIR, SUMMARY_DIR, TABLE_DIR


def _load_compare_tables() -> List[pd.DataFrame]:
    tables = []
    for path in sorted(TABLE_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        if "js_div" not in df.columns:
            continue
        df["compare_name"] = path.stem
        tables.append(df[df["js_div"].notna()].copy())
    return tables


def _load_summary_tables() -> List[pd.DataFrame]:
    tables = []
    for path in sorted(SUMMARY_DIR.glob("*_summary.csv")):
        df = pd.read_csv(path)
        if "anomaly_score" not in df.columns:
            continue
        df["summary_name"] = path.stem
        tables.append(df)
    return tables


def _build_findings_md(top_compare: pd.DataFrame, top_windows: pd.DataFrame) -> str:
    lines = ["# Report Findings (Auto Export)", ""]
    lines.append("## Highest JS-Divergence Clusters")
    lines.append("")
    if top_compare.empty:
        lines.append("- No comparison outputs found.")
    else:
        for _, row in top_compare.iterrows():
            lines.append(
                "- `{0}` cluster {1}: js_div={2:.4f}, reward_gap={3:.4f}, n={4}".format(
                    row["compare_name"],
                    int(row["cluster"]),
                    float(row["js_div"]),
                    float(row["reward_gap"]),
                    int(row["n_total"]),
                )
            )

    lines.append("")
    lines.append("## Anomalous / Regime-Shift Windows")
    lines.append("")
    if top_windows.empty:
        lines.append("- No summary outputs found.")
    else:
        for _, row in top_windows.iterrows():
            lines.append(
                "- `{0}` window k={1}: anomaly={2:.4f}, regime={3:.4f}, r_bar={4:.4f}, switch_rate={5:.4f}".format(
                    row["summary_name"],
                    int(row["k"]),
                    float(row["anomaly_score"]),
                    float(row["regime_shift_score"]),
                    float(row["r_bar"]),
                    float(row["switch_rate"]),
                )
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    compare_tables = _load_compare_tables()
    summary_tables = _load_summary_tables()

    top_compare = pd.concat(compare_tables, ignore_index=True).sort_values("js_div", ascending=False).head(15) if compare_tables else pd.DataFrame()
    top_windows = pd.concat(summary_tables, ignore_index=True).sort_values(["anomaly_score", "regime_shift_score"], ascending=False).head(20) if summary_tables else pd.DataFrame()

    if not top_compare.empty:
        top_compare.to_csv(REPORT_ASSET_DIR / "top_clusters.csv", index=False)
    if not top_windows.empty:
        top_windows.to_csv(REPORT_ASSET_DIR / "top_windows.csv", index=False)

    findings_md = _build_findings_md(top_compare=top_compare, top_windows=top_windows)
    (REPORT_ASSET_DIR / "findings.md").write_text(findings_md, encoding="utf-8")
    print("Saved report assets to: {0}".format(REPORT_ASSET_DIR))


if __name__ == "__main__":
    main()
