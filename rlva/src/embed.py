from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rlva.src.config import SUMMARY_DIR


DEFAULT_FEATURE_COLUMNS: List[str] = [
    "q_bar",
    "mu_bar",
    "sigma_bar",
    "rho_bar",
    "freq0",
    "freq1",
    "freq2",
    "r_bar",
    "H",
]


def _resolve_column(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    if name == "H" and "H_bar" in df.columns:
        return "H_bar"
    raise ValueError(f"Feature column '{name}' not found in summary columns: {list(df.columns)}")


def resolve_feature_columns(df: pd.DataFrame, feature_columns: Optional[Iterable[str]] = None) -> List[str]:
    requested = list(feature_columns) if feature_columns is not None else list(DEFAULT_FEATURE_COLUMNS)
    return [_resolve_column(df, name) for name in requested]


def load_summary_dataframe(
    policy: str,
    summary_path: Optional[Path] = None,
    summaries_dir: Path = SUMMARY_DIR,
) -> pd.DataFrame:
    if summary_path is None:
        summary_path = summaries_dir / f"{policy}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return pd.read_csv(summary_path)


def project_behavior_pca_2d(
    summary_df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, PCA, StandardScaler, List[str]]:
    cols = resolve_feature_columns(summary_df, feature_columns=feature_columns)
    x_raw = summary_df[cols].astype(float)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw.to_numpy())

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    out_df = summary_df.copy()
    out_df["x"] = x_pca[:, 0]
    out_df["y"] = x_pca[:, 1]
    return out_df, pca, scaler, cols
