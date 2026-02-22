"""
deg_rnaseq.py — RNA-seq Differential Expression
Implements simplified DESeq2-style pipeline:
- Negative binomial GLM
- Dispersion estimation
- Wald test
- LFC shrinkage
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def run_rnaseq_deg(count_matrix: pd.DataFrame,
                   group_labels: pd.Series,
                   design: str = "~ group",
                   alpha: float = 0.05) -> pd.DataFrame:
    """
    DESeq2-style RNA-seq DEG analysis.
    
    Parameters:
    - count_matrix: genes × samples (raw counts)
    - group_labels: sample group assignments
    - design: formula (currently only "~ group" supported)
    - alpha: FDR threshold
    
    Returns DataFrame with:
    - baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
    """
    logger.info(f"RNA-seq DEG: {count_matrix.shape}")
    
    # Step 1: Estimate size factors
    size_factors = _estimate_size_factors(count_matrix)
    norm_counts = count_matrix / size_factors
    
    # Step 2: Estimate dispersions
    dispersions = _estimate_dispersions(count_matrix, size_factors, group_labels)
    
    # Step 3: Fit negative binomial GLM for each gene
    results = []
    
    groups = group_labels.unique()
    if len(groups) != 2:
        raise ValueError("Currently only supports 2-group comparison")
    
    group1, group2 = groups
    idx1 = group_labels == group1
    idx2 = group_labels == group2
    
    for gene in count_matrix.index:
        counts = count_matrix.loc[gene].values
        sf = size_factors.values
        disp = dispersions.get(gene, 0.1)
        
        # Mean counts per group
        mean1 = np.mean(counts[idx1] / sf[idx1])
        mean2 = np.mean(counts[idx2] / sf[idx2])
        base_mean = np.mean(counts / sf)
        
        # Log fold change
        log2fc = np.log2((mean2 + 0.5) / (mean1 + 0.5))
        
        # Wald test
        wald_stat, p_val = _wald_test(counts, sf, idx1, idx2, disp)
        
        # Standard error (approximate)
        lfc_se = abs(log2fc / (wald_stat + 1e-10))
        
        results.append({
            "gene": gene,
            "baseMean": round(base_mean, 4),
            "log2FoldChange": round(log2fc, 4),
            "lfcSE": round(lfc_se, 4),
            "stat": round(wald_stat, 4),
            "pvalue": float(p_val),
        })
    
    df = pd.DataFrame(results)
    
    # Step 4: Multiple testing correction
    _, padj, _, _ = multipletests(df["pvalue"].values, method="fdr_bh")
    df["padj"] = padj
    
    # Step 5: LFC shrinkage (optional, simplified)
    df = _shrink_lfc(df)
    
    df["-log10pval"] = -np.log10(df["pvalue"].clip(1e-300))
    df["significant"] = df["padj"] < alpha
    df = df.sort_values("pvalue")
    
    logger.info(f"RNA-seq DEG: {df['significant'].sum()} significant genes")
    return df


def _estimate_size_factors(counts: pd.DataFrame) -> pd.Series:
    """
    DESeq2 median-of-ratios size factor estimation.
    """
    # Geometric mean per gene
    geom_mean = np.exp(np.log(counts.replace(0, np.nan)).mean(axis=1, skipna=True))
    
    # Ratio to geometric mean
    ratios = counts.div(geom_mean, axis=0)
    
    # Median ratio per sample
    size_factors = ratios.median(axis=0)
    
    return size_factors


def _estimate_dispersions(counts: pd.DataFrame,
                          size_factors: pd.Series,
                          group_labels: pd.Series) -> dict:
    """
    Estimate gene-wise dispersions.
    Simplified — real DESeq2 fits a dispersion-mean trend.
    """
    norm_counts = counts / size_factors
    
    dispersions = {}
    for gene in counts.index:
        vals = norm_counts.loc[gene].values
        
        # Variance and mean
        mean_val = np.mean(vals)
        var_val = np.var(vals, ddof=1)
        
        # Negative binomial: var = mean + disp * mean^2
        # Solve for disp
        if mean_val > 0:
            disp = max(0, (var_val - mean_val) / (mean_val ** 2))
        else:
            disp = 0.1
        
        dispersions[gene] = disp
    
    return dispersions


def _wald_test(counts: np.ndarray,
               size_factors: np.ndarray,
               idx1: np.ndarray,
               idx2: np.ndarray,
               dispersion: float) -> tuple:
    """
    Wald test for negative binomial GLM.
    Simplified — real DESeq2 uses iterative fitting.
    """
    # Mean counts per group
    mu1 = np.mean(counts[idx1] / size_factors[idx1])
    mu2 = np.mean(counts[idx2] / size_factors[idx2])
    
    # Standard error (approximate)
    n1 = idx1.sum()
    n2 = idx2.sum()
    
    var1 = mu1 + dispersion * mu1 ** 2
    var2 = mu2 + dispersion * mu2 ** 2
    
    se = np.sqrt(var1 / n1 + var2 / n2)
    
    # Wald statistic
    log2fc = np.log2((mu2 + 0.5) / (mu1 + 0.5))
    wald_stat = log2fc / (se + 1e-10)
    
    # p-value (normal approximation)
    p_val = 2 * stats.norm.sf(abs(wald_stat))
    
    return float(wald_stat), float(p_val)


def _shrink_lfc(df: pd.DataFrame, prior_sd: float = 1.0) -> pd.DataFrame:
    """
    Shrink log fold changes toward zero (apeglm-style).
    Genes with low counts or high variance get shrunk more.
    """
    df = df.copy()
    
    # Shrinkage factor based on SE
    shrinkage = 1 / (1 + (df["lfcSE"] / prior_sd) ** 2)
    
    df["log2FoldChange_shrunken"] = df["log2FoldChange"] * shrinkage
    
    return df


def rnaseq_summary(deg_df: pd.DataFrame, alpha: float = 0.05, lfc_threshold: float = 1.0) -> dict:
    """Generate summary statistics."""
    sig = deg_df[deg_df["padj"] < alpha]
    
    if lfc_threshold > 0:
        sig = sig[sig["log2FoldChange"].abs() >= lfc_threshold]
    
    return {
        "n_total": len(deg_df),
        "n_significant": len(sig),
        "n_up": int((sig["log2FoldChange"] > 0).sum()),
        "n_down": int((sig["log2FoldChange"] < 0).sum()),
        "max_lfc": float(deg_df["log2FoldChange"].abs().max()),
        "min_padj": float(deg_df["padj"].min()),
    }
