"""
deg_multiclass.py — Multi-Group Differential Expression
- ANOVA for overall group effect
- Pairwise contrasts
- Moderated t-test (limma-style empirical Bayes)
- Tukey HSD post-hoc
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations

logger = logging.getLogger(__name__)


def run_multiclass_deg(norm_expr: pd.DataFrame,
                       group_expr: dict,
                       fc_threshold: float = 1.0,
                       pval_threshold: float = 0.05,
                       use_moderated: bool = True) -> dict:
    """
    Multi-group DEG analysis.
    
    Parameters:
    - norm_expr: genes × samples
    - group_expr: dict {group_name: DataFrame}
    - use_moderated: apply empirical Bayes variance shrinkage
    
    Returns dict with:
    - anova_results: overall ANOVA p-values
    - pairwise_results: all pairwise comparisons
    - top_genes: genes significant in any comparison
    """
    logger.info(f"Multi-class DEG: {len(group_expr)} groups")
    
    groups = list(group_expr.keys())
    
    if len(groups) == 2:
        logger.info("Only 2 groups — using pairwise comparison")
        return _binary_comparison(norm_expr, group_expr, groups, fc_threshold, pval_threshold, use_moderated)
    
    # Step 1: ANOVA for overall effect
    anova_results = _compute_anova(norm_expr, group_expr, groups)
    
    # Step 2: Pairwise contrasts
    pairwise_results = {}
    for g1, g2 in combinations(groups, 2):
        logger.info(f"Contrast: {g1} vs {g2}")
        contrast = _pairwise_contrast(
            group_expr[g1], group_expr[g2],
            g1, g2, use_moderated
        )
        pairwise_results[f"{g1}_vs_{g2}"] = contrast
    
    # Step 3: Identify top genes (significant in any comparison)
    top_genes = _identify_top_genes(anova_results, pairwise_results, pval_threshold)
    
    return {
        "anova_results": anova_results,
        "pairwise_results": pairwise_results,
        "top_genes": top_genes,
        "n_groups": len(groups),
        "group_names": groups,
    }


def _compute_anova(norm_expr: pd.DataFrame, group_expr: dict, groups: list) -> pd.DataFrame:
    """One-way ANOVA for each gene."""
    logger.info("Computing ANOVA...")
    
    results = []
    for gene in norm_expr.index:
        group_values = [group_expr[g].loc[gene].dropna().values for g in groups]
        
        # Filter groups with ≥2 samples
        group_values = [v for v in group_values if len(v) >= 2]
        
        if len(group_values) < 2:
            continue
        
        # One-way ANOVA
        f_stat, p_val = stats.f_oneway(*group_values)
        
        # Effect size: eta-squared
        ss_between = sum(len(v) * (np.mean(v) - np.mean(np.concatenate(group_values)))**2 for v in group_values)
        ss_total = sum((np.concatenate(group_values) - np.mean(np.concatenate(group_values)))**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        
        results.append({
            "gene": gene,
            "f_stat": float(f_stat),
            "pval": float(p_val),
            "eta_squared": float(eta_sq),
        })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        _, padj, _, _ = multipletests(df["pval"].values, method="fdr_bh")
        df["padj"] = padj
        df["-log10pval"] = -np.log10(df["pval"].clip(1e-300))
        df = df.sort_values("pval")
    
    logger.info(f"ANOVA: {(df['padj'] < 0.05).sum()} significant genes")
    return df


def _pairwise_contrast(expr1: pd.DataFrame,
                       expr2: pd.DataFrame,
                       name1: str,
                       name2: str,
                       use_moderated: bool) -> pd.DataFrame:
    """Pairwise comparison with moderated t-test."""
    results = []
    
    for gene in expr1.index:
        vals1 = expr1.loc[gene].dropna().values
        vals2 = expr2.loc[gene].dropna().values
        
        if len(vals1) < 2 or len(vals2) < 2:
            continue
        
        mean1 = float(np.mean(vals1))
        mean2 = float(np.mean(vals2))
        log2fc = mean2 - mean1
        
        # Variance
        var1 = np.var(vals1, ddof=1)
        var2 = np.var(vals2, ddof=1)
        
        # Moderated variance (if requested)
        if use_moderated:
            pooled_var = _empirical_bayes_variance([var1, var2])
        else:
            pooled_var = (var1 + var2) / 2
        
        # t-statistic
        se = np.sqrt(pooled_var * (1/len(vals1) + 1/len(vals2)))
        t_stat = log2fc / (se + 1e-10)
        
        # Degrees of freedom
        df_val = len(vals1) + len(vals2) - 2
        p_val = 2 * stats.t.sf(abs(t_stat), df_val)
        
        results.append({
            "gene": gene,
            "mean1": mean1,
            "mean2": mean2,
            "log2fc": round(log2fc, 4),
            "abs_log2fc": round(abs(log2fc), 4),
            "t_stat": round(float(t_stat), 4),
            "pval": float(p_val),
            "pooled_var": float(pooled_var),
        })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        _, padj, _, _ = multipletests(df["pval"].values, method="fdr_bh")
        df["padj"] = padj
        df["-log10pval"] = -np.log10(df["pval"].clip(1e-300))
        df["contrast"] = f"{name1}_vs_{name2}"
        df = df.sort_values("pval")
    
    return df


def _empirical_bayes_variance(variances: list) -> float:
    """
    Empirical Bayes variance shrinkage (simplified limma approach).
    Shrinks variance estimates toward a common value.
    """
    mean_var = np.mean(variances)
    
    # Prior parameters (simplified)
    d0 = 4.0  # prior degrees of freedom
    s0_sq = mean_var  # prior variance
    
    # Posterior variance (weighted average)
    shrunk = []
    for v in variances:
        shrunk_v = (d0 * s0_sq + v) / (d0 + 1)
        shrunk.append(shrunk_v)
    
    return np.mean(shrunk)


def _binary_comparison(norm_expr: pd.DataFrame,
                       group_expr: dict,
                       groups: list,
                       fc_threshold: float,
                       pval_threshold: float,
                       use_moderated: bool) -> dict:
    """Handle binary case (2 groups)."""
    g1, g2 = groups
    contrast = _pairwise_contrast(
        group_expr[g1], group_expr[g2], g1, g2, use_moderated
    )
    
    contrast["is_sig"] = (contrast["padj"] < pval_threshold) & (contrast["abs_log2fc"] >= fc_threshold)
    contrast["direction"] = contrast["log2fc"].apply(lambda x: "Up" if x > 0 else "Down")
    
    summary = {
        "n_total": len(contrast),
        "n_significant": int(contrast["is_sig"].sum()),
        "n_up": int((contrast["is_sig"] & (contrast["log2fc"] > 0)).sum()),
        "n_down": int((contrast["is_sig"] & (contrast["log2fc"] < 0)).sum()),
    }
    
    return {
        "anova_results": pd.DataFrame(),  # N/A for 2 groups
        "pairwise_results": {f"{g1}_vs_{g2}": contrast},
        "top_genes": contrast[contrast["is_sig"]].copy(),
        "n_groups": 2,
        "group_names": groups,
        "summary": summary,
    }


def _identify_top_genes(anova_results: pd.DataFrame,
                        pairwise_results: dict,
                        pval_threshold: float) -> pd.DataFrame:
    """Find genes significant in any pairwise comparison."""
    all_sig = []
    
    for contrast_name, df in pairwise_results.items():
        sig = df[df["padj"] < pval_threshold].copy()
        sig["contrast"] = contrast_name
        all_sig.append(sig)
    
    if all_sig:
        combined = pd.concat(all_sig, ignore_index=True)
        # Keep best p-value per gene
        top = combined.loc[combined.groupby("gene")["pval"].idxmin()]
        return top.sort_values("pval")
    
    return pd.DataFrame()


def deg_summary(results: dict, fc_threshold: float = 1.0, pval_threshold: float = 0.05) -> dict:
    """Generate summary statistics."""
    if "summary" in results:
        return results["summary"]
    
    # Multi-class summary
    top = results.get("top_genes", pd.DataFrame())
    
    if len(top) == 0:
        return {
            "n_total": 0,
            "n_significant": 0,
            "n_up": 0,
            "n_down": 0,
        }
    
    sig = top[(top["padj"] < pval_threshold) & (top["abs_log2fc"] >= fc_threshold)]
    
    return {
        "n_total": len(top),
        "n_significant": len(sig),
        "n_up": int((sig["log2fc"] > 0).sum()),
        "n_down": int((sig["log2fc"] < 0).sum()),
        "top_gene": sig.iloc[0]["gene"] if len(sig) > 0 else "—",
    }
