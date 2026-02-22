"""
pca_analysis.py — Real PCA, hierarchical clustering, and dimensionality reduction
Uses scikit-learn for PCA, scipy for clustering + dendrogram.
"""

import logging
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────────────────────────

def run_pca(norm_expr: pd.DataFrame,
            sample_info: pd.DataFrame,
            n_components: int = 10,
            top_var_genes: int = 1000) -> dict:
    """
    Run PCA on expression matrix.
    Uses top variable genes to reduce noise.

    Returns dict with coordinates, loadings, variance explained.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Select top variable genes
    gene_var = norm_expr.var(axis=1).sort_values(ascending=False)
    top_genes = gene_var.head(min(top_var_genes, len(gene_var))).index
    sub = norm_expr.loc[top_genes].T.fillna(0)  # samples × genes

    # Standardize
    X = StandardScaler().fit_transform(sub)

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X)

    # Sample coordinates DataFrame
    coord_df = pd.DataFrame(
        coords,
        index=norm_expr.columns,
        columns=[f"PC{i+1}" for i in range(n_comp)]
    )
    coord_df["group"] = coord_df.index.map(
        lambda s: sample_info.loc[s, "group"] if s in sample_info.index else "Unknown"
    )
    coord_df["sample"] = coord_df.index

    # Gene loadings (top contributors to each PC)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=top_genes,
        columns=[f"PC{i+1}" for i in range(n_comp)]
    )

    # Top genes per PC
    top_loadings = {}
    for pc in [f"PC{i+1}" for i in range(min(3, n_comp))]:
        top_pos = loadings[pc].nlargest(10)
        top_neg = loadings[pc].nsmallest(10)
        top_loadings[pc] = {
            "positive": top_pos.to_dict(),
            "negative": top_neg.to_dict(),
        }

    variance_explained = (pca.explained_variance_ratio_ * 100).round(2).tolist()
    cumulative_var     = np.cumsum(pca.explained_variance_ratio_ * 100).round(2).tolist()

    logger.info(f"PCA: PC1={variance_explained[0]:.1f}%, PC2={variance_explained[1]:.1f}% variance explained")

    return {
        "coordinates":       coord_df,
        "loadings":          loadings,
        "top_loadings":      top_loadings,
        "variance_explained": variance_explained,
        "cumulative_var":    cumulative_var,
        "n_components":      n_comp,
        "n_genes_used":      len(top_genes),
        "pca_object":        pca,
    }


# ──────────────────────────────────────────────────────────────────
# HIERARCHICAL CLUSTERING
# ──────────────────────────────────────────────────────────────────

def run_hierarchical_clustering(norm_expr: pd.DataFrame,
                                 sample_info: pd.DataFrame,
                                 method: str = "complete",
                                 metric: str = "euclidean",
                                 max_samples: int = 50) -> dict:
    """
    Run hierarchical clustering on samples.

    Parameters
    ----------
    norm_expr    : genes × samples
    method       : 'complete', 'average', 'ward', 'single'
    metric       : 'euclidean', 'correlation', 'cosine'
    """
    samples = norm_expr.columns[:max_samples].tolist()
    X = norm_expr[samples].T.fillna(0).values  # samples × genes

    if metric == "correlation":
        dist_condensed = pdist(X, metric="correlation")
    elif metric == "cosine":
        dist_condensed = pdist(X, metric="cosine")
    else:
        dist_condensed = pdist(X, metric="euclidean")

    # Handle ward linkage (requires euclidean)
    if method == "ward":
        link = linkage(X, method="ward")
    else:
        link = linkage(dist_condensed, method=method)

    # Distance matrix (full)
    dist_matrix = squareform(dist_condensed)
    dist_df = pd.DataFrame(dist_matrix, index=samples, columns=samples)

    # Cut tree to get flat clusters (2 clusters default)
    n_clusters = 2
    flat_clusters = fcluster(link, t=n_clusters, criterion="maxclust")
    cluster_labels = {s: int(c) for s, c in zip(samples, flat_clusters)}

    # Dendrogram data for JS rendering
    dendro = dendrogram(link, labels=samples, no_plot=True)

    return {
        "linkage_matrix": link,
        "distance_matrix": dist_df,
        "dendro_data": {
            "icoord":   dendro["icoord"],
            "dcoord":   dendro["dcoord"],
            "ivl":      dendro["ivl"],
            "leaves":   dendro["leaves"],
            "color_list": dendro["color_list"],
        },
        "cluster_labels": cluster_labels,
        "samples": samples,
        "method": method,
        "metric": metric,
    }


# ──────────────────────────────────────────────────────────────────
# GENE CORRELATION
# ──────────────────────────────────────────────────────────────────

def compute_gene_correlation(norm_expr: pd.DataFrame,
                              top_n: int = 40,
                              method: str = "pearson") -> dict:
    """Pearson/Spearman correlation between top variable genes."""
    gene_var = norm_expr.var(axis=1).sort_values(ascending=False)
    top_genes = gene_var.head(top_n).index.tolist()
    sub = norm_expr.loc[top_genes]

    if method == "spearman":
        corr_matrix = sub.T.corr(method="spearman")
    else:
        corr_matrix = sub.T.corr(method="pearson")

    corr_matrix = corr_matrix.round(4)

    # Find top co-expressed pairs
    pairs = []
    n = len(top_genes)
    corr_arr = corr_matrix.values
    for i in range(n):
        for j in range(i+1, n):
            r = float(corr_arr[i, j])
            pairs.append({
                "gene1": top_genes[i],
                "gene2": top_genes[j],
                "r":     round(r, 4),
                "abs_r": round(abs(r), 4),
            })
    pairs.sort(key=lambda x: x["abs_r"], reverse=True)

    return {
        "corr_matrix":    corr_matrix,
        "genes":          top_genes,
        "top_pairs":      pairs[:30],
        "method":         method,
    }


# ──────────────────────────────────────────────────────────────────
# SAMPLE-SAMPLE CORRELATION
# ──────────────────────────────────────────────────────────────────

def compute_sample_correlation(norm_expr: pd.DataFrame,
                                max_samples: int = 30) -> pd.DataFrame:
    """Pearson correlation between all pairs of samples."""
    samples = norm_expr.columns[:max_samples].tolist()
    sub = norm_expr[samples]
    return sub.corr(method="pearson").round(4)
