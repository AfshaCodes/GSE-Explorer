"""
preprocessor.py — Real expression data preprocessing
Log2 normalization, quantile normalization, low-expression filtering,
outlier detection, and QC metrics.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import median_abs_deviation

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────

def preprocess(expr_df: pd.DataFrame,
               sample_info: pd.DataFrame,
               log2_transform: bool = True,
               normalize: str = "quantile",
               filter_low: bool = True,
               min_expr: float = 1.0,
               min_samples_frac: float = 0.2) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    expr_df          : genes × samples raw expression DataFrame
    sample_info      : sample metadata with 'group' column
    log2_transform   : whether to apply log2(x + 1)
    normalize        : 'quantile', 'zscore', 'median_center', or None
    filter_low       : remove genes with low expression
    min_expr         : minimum expression threshold
    min_samples_frac : fraction of samples that must exceed min_expr

    Returns dict with processed data + QC stats.
    """
    logger.info(f"Preprocessing: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")

    raw = expr_df.copy().astype(float)

    # 1. Remove all-zero / all-NaN genes
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.dropna(how="all")
    raw = raw.loc[~(raw == 0).all(axis=1)]
    logger.info(f"After NaN/zero removal: {raw.shape[0]} genes")

    # 2. Low-expression filter
    if filter_low:
        min_samples = max(1, int(raw.shape[1] * min_samples_frac))
        keep = (raw > min_expr).sum(axis=1) >= min_samples
        raw = raw.loc[keep]
        logger.info(f"After low-expr filter: {raw.shape[0]} genes")

    # 3. Log2 transform (if not already log-scaled)
    if log2_transform:
        if _is_already_log(raw):
            logger.info("Data appears already log-scaled, skipping log2 transform")
            log_expr = raw
        else:
            log_expr = np.log2(raw + 1)
    else:
        log_expr = raw

    # 4. Normalization
    if normalize == "quantile":
        norm_expr = _quantile_normalize(log_expr)
    elif normalize == "zscore":
        norm_expr = _zscore_normalize(log_expr)
    elif normalize == "median_center":
        norm_expr = _median_center(log_expr)
    else:
        norm_expr = log_expr

    # 5. Outlier sample detection
    outliers = _detect_outlier_samples(norm_expr)

    # 6. QC metrics
    qc = _compute_qc_metrics(raw, log_expr, norm_expr, sample_info)

    # 7. Per-gene stats
    gene_stats = _compute_gene_stats(norm_expr)

    return {
        "raw":        raw,
        "log_expr":   log_expr,
        "norm_expr":  norm_expr,
        "gene_stats": gene_stats,
        "qc_metrics": qc,
        "outlier_samples": outliers,
        "n_genes_raw": expr_df.shape[0],
        "n_genes_filtered": norm_expr.shape[0],
        "n_samples": norm_expr.shape[1],
        "normalization": normalize,
    }


# ──────────────────────────────────────────────────────────────────
# NORMALIZATION METHODS
# ──────────────────────────────────────────────────────────────────

def _is_already_log(df: pd.DataFrame, threshold: float = 20.0) -> bool:
    """Heuristic: if max value < threshold, likely already log-scaled."""
    return float(df.max().max()) < threshold


def _quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full quantile normalization:
    1. Rank each column
    2. Sort each column, compute row means
    3. Replace ranks with means
    """
    arr = df.values.copy().astype(float)
    n_genes, n_samples = arr.shape

    # Step 1: rank (argsort of argsort)
    ranks = np.argsort(np.argsort(arr, axis=0), axis=0)

    # Step 2: sort each column and compute row-wise mean
    sorted_arr = np.sort(arr, axis=0)
    row_means  = sorted_arr.mean(axis=1)

    # Step 3: replace with means
    result = np.zeros_like(arr)
    for j in range(n_samples):
        result[:, j] = row_means[ranks[:, j]]

    return pd.DataFrame(result, index=df.index, columns=df.columns)


def _zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-score normalization."""
    return df.apply(lambda col: (col - col.mean()) / (col.std() + 1e-8), axis=0)


def _median_center(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract column median — preserves relative differences."""
    return df.sub(df.median(axis=0), axis=1)


# ──────────────────────────────────────────────────────────────────
# QC METRICS
# ──────────────────────────────────────────────────────────────────

def _compute_qc_metrics(raw: pd.DataFrame,
                         log_expr: pd.DataFrame,
                         norm_expr: pd.DataFrame,
                         sample_info: pd.DataFrame) -> dict:
    """Compute per-sample and global QC metrics."""
    per_sample = []
    for col in norm_expr.columns:
        vals = norm_expr[col].dropna().values
        raw_vals = raw[col].dropna().values if col in raw.columns else vals
        grp = sample_info.loc[col, "group"] if col in sample_info.index else "Unknown"
        per_sample.append({
            "sample":      col,
            "group":       grp,
            "mean_expr":   float(np.mean(vals)),
            "median_expr": float(np.median(vals)),
            "std_expr":    float(np.std(vals)),
            "q1":          float(np.percentile(vals, 25)),
            "q3":          float(np.percentile(vals, 75)),
            "iqr":         float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "n_detected":  int((raw_vals > 0).sum()),
            "detection_rate": float((raw_vals > 0).mean() * 100),
        })

    sample_df = pd.DataFrame(per_sample)

    # Inter-sample correlation matrix
    corr_matrix = norm_expr.corr(method="pearson")

    # Global stats
    all_vals = norm_expr.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]

    global_stats = {
        "global_mean":   float(np.mean(all_vals)),
        "global_median": float(np.median(all_vals)),
        "global_std":    float(np.std(all_vals)),
        "global_min":    float(np.min(all_vals)),
        "global_max":    float(np.max(all_vals)),
        "n_genes":       int(norm_expr.shape[0]),
        "n_samples":     int(norm_expr.shape[1]),
        "inter_sample_corr_mean": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
    }

    return {
        "per_sample":   sample_df.to_dict(orient="records"),
        "corr_matrix":  corr_matrix,
        "global":       global_stats,
    }


# ──────────────────────────────────────────────────────────────────
# GENE STATS
# ──────────────────────────────────────────────────────────────────

def _compute_gene_stats(norm_expr: pd.DataFrame) -> pd.DataFrame:
    """Per-gene summary statistics."""
    df = pd.DataFrame({
        "mean":     norm_expr.mean(axis=1),
        "median":   norm_expr.median(axis=1),
        "std":      norm_expr.std(axis=1),
        "variance": norm_expr.var(axis=1),
        "cv":       norm_expr.std(axis=1) / (norm_expr.mean(axis=1).abs() + 1e-8),
        "min":      norm_expr.min(axis=1),
        "max":      norm_expr.max(axis=1),
        "mad":      norm_expr.apply(lambda r: float(median_abs_deviation(r.dropna())), axis=1),
    })
    df["range"] = df["max"] - df["min"]
    return df.sort_values("variance", ascending=False)


# ──────────────────────────────────────────────────────────────────
# OUTLIER DETECTION
# ──────────────────────────────────────────────────────────────────

def _detect_outlier_samples(norm_expr: pd.DataFrame,
                              n_sd: float = 3.0) -> list:
    """
    Detect outlier samples using PCA distance from centroid.
    Samples > n_sd standard deviations from mean PC1/PC2 are flagged.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = norm_expr.T.fillna(0).values
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(2, X.shape[1]))
        coords = pca.fit_transform(X)

        center  = coords.mean(axis=0)
        dists   = np.linalg.norm(coords - center, axis=1)
        threshold = dists.mean() + n_sd * dists.std()
        outliers = [norm_expr.columns[i] for i, d in enumerate(dists) if d > threshold]
        return outliers
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────────

def split_by_group(norm_expr: pd.DataFrame,
                   sample_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split expression matrix into two groups for binary DEG analysis.
    Works with any group names — not just "Control"/"Treatment".

    Logic:
    - If "Control"/"Treatment" labels exist, use them.
    - Otherwise use the first two unique group labels (alphabetically for
      reproducibility; for timepoints the caller should use split_by_all_groups).
    - Samples whose group label doesn't map to either side are dropped.
    """
    unique_groups = sample_info["group"].unique().tolist()

    # Determine the two group slots
    if "Control" in unique_groups and "Treatment" in unique_groups:
        g1_label, g2_label = "Control", "Treatment"
    else:
        # Sort for reproducibility; timepoint series are handled by split_by_all_groups
        sorted_groups = sorted(unique_groups)
        g1_label = sorted_groups[0]
        g2_label = sorted_groups[1] if len(sorted_groups) > 1 else sorted_groups[0]

    g1_samples = sample_info[sample_info["group"] == g1_label].index.tolist()
    g2_samples = sample_info[sample_info["group"] == g2_label].index.tolist()

    g1_samples = [s for s in g1_samples if s in norm_expr.columns]
    g2_samples = [s for s in g2_samples if s in norm_expr.columns]

    logger.info(f"Binary split: '{g1_label}' ({len(g1_samples)}) vs '{g2_label}' ({len(g2_samples)})")

    return norm_expr[g1_samples], norm_expr[g2_samples]


def split_by_all_groups(norm_expr: pd.DataFrame,
                        sample_info: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split expression matrix by ALL unique group labels.

    Returns
    -------
    dict mapping group_label → DataFrame (genes × group_samples)
    Order is preserved: for time-series data the dict is ordered by timepoint.
    """
    import re

    groups = sample_info["group"].unique().tolist()

    # If groups look like timepoints (T_0, T_24h …) sort them numerically
    def _tp_key(g: str):
        m = re.search(r"(\d+(?:\.\d+)?)", g)
        return float(m.group(1)) if m else 0.0

    if all(str(g).startswith("T_") for g in groups):
        groups = sorted(groups, key=_tp_key)

    result = {}
    for grp in groups:
        samples = sample_info[sample_info["group"] == grp].index.tolist()
        samples = [s for s in samples if s in norm_expr.columns]
        if samples:
            result[grp] = norm_expr[samples]

    return result
