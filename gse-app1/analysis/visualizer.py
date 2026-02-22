"""
visualizer.py — Real matplotlib/seaborn plot generation
All functions return base64-encoded PNG strings for serving via Flask.
"""

import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

logger = logging.getLogger(__name__)

# ── Global style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1420",
    "axes.facecolor":    "#0d1420",
    "axes.edgecolor":    "#1f3050",
    "axes.labelcolor":   "#6b8ab0",
    "xtick.color":       "#4a6080",
    "ytick.color":       "#4a6080",
    "text.color":        "#e8edf5",
    "grid.color":        "#1f3050",
    "grid.linewidth":    0.5,
    "axes.grid":         True,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "font.family":       "monospace",
    "figure.dpi":        110,
})

CYAN   = "#00f0d4"
VIOLET = "#8b5cf6"
GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"
BLUE   = "#3b82f6"
MUTED  = "#4a6080"
SUBTLE = "#6b8ab0"

GROUP_COLORS = {"Control": CYAN, "Treatment": VIOLET, "Unknown": MUTED}

# Extended palette for multi-class/time-series groups
_EXTRA_COLORS = [
    "#00f0d4", "#8b5cf6", "#f59e0b", "#3b82f6", "#ec4899",
    "#10b981", "#f97316", "#a78bfa", "#14b8a6", "#e11d48",
    "#84cc16", "#0ea5e9", "#fb923c", "#c084fc", "#34d399",
]


def _dynamic_group_colors(groups: list) -> dict:
    """
    Return a colour mapping for any set of group labels.
    Known labels get their fixed colour; unknown labels get
    distinct colours from the extended palette.
    """
    mapping = {}
    extra_idx = 0
    for g in groups:
        if g in GROUP_COLORS:
            mapping[g] = GROUP_COLORS[g]
        else:
            mapping[g] = _EXTRA_COLORS[extra_idx % len(_EXTRA_COLORS)]
            extra_idx += 1
    return mapping


def _group_palette(sample_info: pd.DataFrame) -> dict:
    groups  = sample_info["group"].unique().tolist() if "group" in sample_info.columns else []
    gcolors = _dynamic_group_colors(groups)
    colors  = {}
    for s in sample_info.index:
        grp = sample_info.loc[s, "group"] if "group" in sample_info.columns else "Unknown"
        colors[s] = gcolors.get(grp, MUTED)
    return colors


def _to_b64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


# ──────────────────────────────────────────────────────────────────
# 1. SAMPLE BOX PLOT (per-sample expression distribution)
# ──────────────────────────────────────────────────────────────────

def plot_sample_boxplot(norm_expr: pd.DataFrame,
                         sample_info: pd.DataFrame) -> str:
    n = min(norm_expr.shape[1], 40)
    samples = norm_expr.columns[:n].tolist()
    palette = _group_palette(sample_info)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), 5))
    data_list = [norm_expr[s].dropna().values for s in samples]
    colors    = [palette.get(s, MUTED) for s in samples]

    bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, n+1))
    ax.set_xticklabels([s[-6:] for s in samples], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("log₂ Expression", fontsize=9)
    ax.set_title("Per-Sample Expression Distribution (Normalized)", fontsize=11)

    # Legend
    present_groups = sample_info["group"].unique().tolist() if "group" in sample_info.columns else []
    gcolors = _dynamic_group_colors(present_groups)
    handles = [mpatches.Patch(color=gcolors[g], label=g) for g in present_groups]
    ax.legend(handles=handles, loc="upper right", framealpha=0.2, fontsize=8)
    fig.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────────
# 2. VOLCANO PLOT
# ──────────────────────────────────────────────────────────────────

def plot_volcano(deg_df: pd.DataFrame,
                 fc_thresh: float = 1.0,
                 pval_thresh: float = 0.05,
                 label_top: int = 12) -> str:
    fig, ax = plt.subplots(figsize=(9, 7))

    # ANOVA result has no log2fc — render F-statistic plot instead
    if "log2fc" not in deg_df.columns:
        sig = deg_df[deg_df["is_sig_deg"]]
        ns  = deg_df[~deg_df["is_sig_deg"]]
        ax.scatter(ns["max_log2fc"],  ns["-log10pval"], c=MUTED,  alpha=0.35, s=12, linewidths=0, label=f"NS ({len(ns)})")
        ax.scatter(sig["max_log2fc"], sig["-log10pval"], c=AMBER, alpha=0.75, s=18, linewidths=0, label=f"Sig ({len(sig)})")
        ax.axhline(-np.log10(pval_thresh), color=SUBTLE, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(fc_thresh, color=SUBTLE, linestyle="--", linewidth=0.8, alpha=0.6)
        top = deg_df[deg_df["is_sig_deg"]].nlargest(label_top, "max_log2fc")
        for _, row in top.iterrows():
            ax.annotate(row["gene"], (row["max_log2fc"], row["-log10pval"]),
                        fontsize=6.5, color="#cbd5e1", xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("Max |log₂FC| across groups")
        ax.set_ylabel("−log₁₀(p-value, ANOVA)")
        ax.set_title("ANOVA Significance Plot (multi-group)")
        ax.legend(framealpha=0.15, fontsize=8)
        fig.tight_layout()
        return _to_b64(fig)

    ns  = deg_df[~deg_df["is_sig_deg"]]
    up  = deg_df[deg_df["is_sig_deg"] & (deg_df["log2fc"] > 0)]
    dn  = deg_df[deg_df["is_sig_deg"] & (deg_df["log2fc"] < 0)]

    ax.scatter(ns["log2fc"],  ns["-log10pval"],  c=MUTED,  alpha=0.35, s=12, linewidths=0, label=f"NS ({len(ns)})")
    ax.scatter(up["log2fc"],  up["-log10pval"],  c=GREEN,  alpha=0.75, s=18, linewidths=0, label=f"Up ({len(up)})")
    ax.scatter(dn["log2fc"],  dn["-log10pval"],  c=RED,    alpha=0.75, s=18, linewidths=0, label=f"Down ({len(dn)})")

    # Threshold lines
    pval_line = -np.log10(pval_thresh)
    ax.axhline(pval_line, color=SUBTLE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline( fc_thresh, color=SUBTLE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(-fc_thresh, color=SUBTLE, linestyle="--", linewidth=0.8, alpha=0.6)

    # Labels for top DEGs
    top = deg_df[deg_df["is_sig_deg"]].nlargest(label_top, "abs_log2fc")
    for _, row in top.iterrows():
        ax.annotate(row["gene"],
                    (row["log2fc"], row["-log10pval"]),
                    fontsize=6.5, color="#cbd5e1",
                    xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("log₂ Fold Change (Treatment / Control)")
    ax.set_ylabel("−log₁₀(p-value)")
    ax.set_title("Volcano Plot — Differential Expression")
    ax.legend(framealpha=0.15, fontsize=8)
    ax.text(0.99, 0.01, f"|FC|≥{fc_thresh}, padj<{pval_thresh}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=SUBTLE)
    fig.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────────
# 3. EXPRESSION HEATMAP (top variable genes)
# ──────────────────────────────────────────────────────────────────

def plot_heatmap(norm_expr: pd.DataFrame,
                 sample_info: pd.DataFrame,
                 n_genes: int = 50,
                 n_samples: int = 30) -> str:
    gene_var  = norm_expr.var(axis=1).sort_values(ascending=False)
    top_genes = gene_var.head(n_genes).index.tolist()
    samples   = norm_expr.columns[:n_samples].tolist()
    sub = norm_expr.loc[top_genes, samples]

    # Group-ordered column colors
    present_groups = [sample_info.loc[s, "group"] if s in sample_info.index else "Unknown" for s in samples]
    gcolors = _dynamic_group_colors(list(dict.fromkeys(present_groups)))  # preserve order, dedupe
    grp_colors = [gcolors.get(g, MUTED) for g in present_groups]

    row_colors = None
    col_colors = pd.Series(grp_colors, index=samples)

    fig_h = max(8, n_genes * 0.22)
    fig_w = max(10, n_samples * 0.25)

    g = sns.clustermap(
        sub,
        cmap="RdBu_r",
        center=sub.values.mean(),
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=True,
        figsize=(fig_w, fig_h),
        xticklabels=[s[-6:] for s in samples],
        yticklabels=top_genes,
        cbar_kws={"label": "log₂ expression"},
        linewidths=0,
        tree_kws={"linewidths": 0.5, "colors": [SUBTLE]},
    )
    g.fig.patch.set_facecolor("#0d1420")
    g.ax_heatmap.set_facecolor("#0d1420")
    g.ax_heatmap.tick_params(labelsize=6.5)
    g.fig.suptitle("Expression Heatmap — Top Variable Genes", y=1.01, fontsize=11, color="#e8edf5")

    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0d1420")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(g.fig)
    return f"data:image/png;base64,{encoded}"


# ──────────────────────────────────────────────────────────────────
# 4. PCA PLOT
# ──────────────────────────────────────────────────────────────────

def plot_pca(pca_result: dict, sample_info: pd.DataFrame) -> str:
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    coord_df = pca_result["coordinates"]
    ve       = pca_result["variance_explained"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── PC1 vs PC2 ──
    ax = axes[0]
    groups = coord_df["group"].unique().tolist()
    gcolors = _dynamic_group_colors(groups)
    for grp in groups:
        sub = coord_df[coord_df["group"] == grp]
        c   = gcolors.get(grp, MUTED)
        ax.scatter(sub["PC1"], sub["PC2"], c=c, s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=grp, zorder=3)

        # 95% confidence ellipse
        if len(sub) > 2:
            x, y   = sub["PC1"].values, sub["PC2"].values
            mx, my = x.mean(), y.mean()
            sx, sy = x.std(), y.std()
            ellipse = Ellipse((mx, my), width=sx*4, height=sy*4,
                              angle=0, facecolor=c, alpha=0.08,
                              edgecolor=c, linewidth=1.0, linestyle="--")
            ax.add_patch(ellipse)
            ax.text(mx, my + sy*2 + 0.3, grp, ha="center", fontsize=8, color=c, alpha=0.7)

    ax.set_xlabel(f"PC1 ({ve[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ve[1]:.1f}% variance)")
    ax.set_title("PCA — PC1 vs PC2")
    ax.legend(framealpha=0.15, fontsize=8)

    # ── Scree plot ──
    ax2 = axes[1]
    n_shown = min(10, len(ve))
    x_pos = range(1, n_shown + 1)
    cumve = np.cumsum(ve[:n_shown])

    bars = ax2.bar(x_pos, ve[:n_shown], color=CYAN, alpha=0.65, width=0.6, label="Individual")
    ax2.plot(x_pos, cumve, color=AMBER, marker="o", markersize=5, linewidth=2, label="Cumulative")
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels([f"PC{i}" for i in x_pos])
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Variance Explained (%)")
    ax2.set_title("Scree Plot")
    ax2.legend(framealpha=0.15, fontsize=8)
    ax2.axhline(y=1, color=SUBTLE, linestyle=":", linewidth=0.8)

    fig.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────────
# 5. VIOLIN / DISTRIBUTION PLOT
# ──────────────────────────────────────────────────────────────────

def plot_violin(norm_expr: pd.DataFrame,
                sample_info: pd.DataFrame) -> str:
    ctrl_s  = sample_info[sample_info["group"] == "Control"].index.tolist()
    treat_s = sample_info[sample_info["group"] == "Treatment"].index.tolist()
    ctrl_s  = [s for s in ctrl_s  if s in norm_expr.columns]
    treat_s = [s for s in treat_s if s in norm_expr.columns]

    ctrl_vals  = norm_expr[ctrl_s].values.flatten()
    treat_vals = norm_expr[treat_s].values.flatten()
    ctrl_vals  = ctrl_vals[~np.isnan(ctrl_vals)]
    treat_vals = treat_vals[~np.isnan(treat_vals)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violin
    ax = axes[0]
    parts = ax.violinplot([ctrl_vals, treat_vals], positions=[1, 2],
                          showmedians=True, showextrema=True)
    for i, (pc, color) in enumerate(zip(parts["bodies"], [CYAN, VIOLET])):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    parts["cbars"].set_color(SUBTLE)
    parts["cmins"].set_color(SUBTLE)
    parts["cmaxes"].set_color(SUBTLE)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Control", "Treatment"])
    ax.set_ylabel("log₂ Expression")
    ax.set_title("Expression Violin Plot")

    # QQ plot (Control group vs Normal)
    ax2 = axes[1]
    from scipy.stats import probplot
    (osm, osr), (slope, intercept, r) = probplot(ctrl_vals[:5000], dist="norm")
    ax2.scatter(osm, osr, c=CYAN, s=6, alpha=0.5, linewidths=0)
    ax2.plot(osm, slope*np.array(osm)+intercept, c=AMBER, linewidth=1.5, label=f"r={r:.3f}")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("QQ Plot — Control Group Normality")
    ax2.legend(framealpha=0.15, fontsize=8)

    fig.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────────
# 6. CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                              title: str = "Gene Correlation Matrix") -> str:
    n = len(corr_matrix)
    fig_size = max(8, n * 0.3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.zeros_like(corr_matrix.values, dtype=bool)

    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        square=True,
        linewidths=0.3,
        linecolor="#0d1420",
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.index,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        annot=(n <= 20),
        fmt=".2f" if n <= 20 else "",
        annot_kws={"size": 6.5} if n <= 20 else {},
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(labelsize=7.5, axis="both")
    fig.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────────
# 7. PATHWAY ENRICHMENT PLOT
# ──────────────────────────────────────────────────────────────────

def plot_enrichment(enrich_df: pd.DataFrame, top_n: int = 15) -> str:
    if enrich_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No significant pathways found", ha="center", va="center",
                fontsize=12, color=SUBTLE)
        return _to_b64(fig)

    top = enrich_df.head(top_n).copy()
    top["-log10pval"] = -np.log10(top["pval"].clip(1e-30))
    top = top.sort_values("-log10pval", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Bar chart ──
    ax = axes[0]
    colors = [GREEN if sig else MUTED for sig in top["significant"]]
    bars = ax.barh(top["pathway"], top["-log10pval"], color=colors, alpha=0.75, edgecolor="none")
    ax.axvline(-np.log10(0.05), color=AMBER, linestyle="--", linewidth=1, alpha=0.7, label="p=0.05")
    ax.set_xlabel("−log₁₀(p-value)")
    ax.set_title("Pathway Enrichment (ORA)")
    ax.legend(framealpha=0.15, fontsize=8)
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(0.05, i, f"{row['n_overlap']}/{row['n_pathway_genes']}", va="center", fontsize=6.5, color="white", alpha=0.8)

    # ── Bubble chart ──
    ax2 = axes[1]
    sc = ax2.scatter(
        top["fold_enrichment"],
        top["-log10pval"],
        s=[max(20, r*400) for r in top["gene_ratio"]],
        c=top["-log10pval"],
        cmap="plasma",
        alpha=0.8,
        edgecolors=SUBTLE,
        linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax2, label="−log₁₀(p-value)", shrink=0.7)
    ax2.axhline(-np.log10(0.05), color=AMBER, linestyle="--", linewidth=0.8, alpha=0.6)
    for _, row in top.iterrows():
        ax2.annotate(row["pathway"][:22],
                     (row["fold_enrichment"], row["-log10pval"]),
                     fontsize=6, color="#cbd5e1",
                     xytext=(4, 2), textcoords="offset points")
    ax2.set_xlabel("Fold Enrichment")
    ax2.set_ylabel("−log₁₀(p-value)")
    ax2.set_title("Pathway Bubble Plot")

    fig.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────────
# 8. MA PLOT
# ──────────────────────────────────────────────────────────────────

def plot_ma(deg_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))

    # ANOVA result has no log2fc — use max_log2fc as y-axis instead
    if "log2fc" not in deg_df.columns:
        sig = deg_df[deg_df["is_sig_deg"]]
        ns  = deg_df[~deg_df["is_sig_deg"]]
        ax.scatter(ns["base_mean"],  ns["max_log2fc"],  c=MUTED, s=8, alpha=0.3, linewidths=0)
        ax.scatter(sig["base_mean"], sig["max_log2fc"], c=AMBER, s=14, alpha=0.75, linewidths=0, label=f"Sig ANOVA ({len(sig)})")
        ax.axhline(0, color=SUBTLE, linewidth=0.8)
        top = deg_df[deg_df["is_sig_deg"]].nlargest(8, "max_log2fc")
        for _, row in top.iterrows():
            ax.annotate(row["gene"], (row["base_mean"], row["max_log2fc"]),
                        fontsize=6.5, color="#cbd5e1", xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("Average log₂ Expression (A)")
        ax.set_ylabel("Max |log₂FC| across groups (M)")
        ax.set_title("MA Plot — ANOVA (multi-group)")
        ax.legend(framealpha=0.15, fontsize=8)
        fig.tight_layout()
        return _to_b64(fig)

    ns = deg_df[~deg_df["is_sig_deg"]]
    up = deg_df[deg_df["is_sig_deg"] & (deg_df["log2fc"] > 0)]
    dn = deg_df[deg_df["is_sig_deg"] & (deg_df["log2fc"] < 0)]

    ax.scatter(ns["base_mean"], ns["log2fc"], c=MUTED, s=8, alpha=0.3, linewidths=0)
    ax.scatter(up["base_mean"], up["log2fc"], c=GREEN, s=14, alpha=0.75, linewidths=0, label=f"Up ({len(up)})")
    ax.scatter(dn["base_mean"], dn["log2fc"], c=RED,   s=14, alpha=0.75, linewidths=0, label=f"Down ({len(dn)})")
    ax.axhline(0, color=SUBTLE, linewidth=0.8)
    ax.axhline(1,  color=SUBTLE, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(-1, color=SUBTLE, linestyle="--", linewidth=0.6, alpha=0.5)

    top = deg_df[deg_df["is_sig_deg"]].nlargest(8, "abs_log2fc")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["base_mean"], row["log2fc"]),
                    fontsize=6.5, color="#cbd5e1",
                    xytext=(3, 2), textcoords="offset points")

    ax.set_xlabel("Average log₂ Expression (A)")
    ax.set_ylabel("log₂ Fold Change (M)")
    ax.set_title("MA Plot (Bland–Altman)")
    ax.legend(framealpha=0.15, fontsize=8)
    fig.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────────
# 9. DENSITY PLOT (per-sample expression density)
# ──────────────────────────────────────────────────────────────────

def plot_density(norm_expr: pd.DataFrame,
                 sample_info: pd.DataFrame,
                 max_samples: int = 10) -> str:
    from scipy.stats import gaussian_kde

    samples = norm_expr.columns[:max_samples].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))

    present_groups = sample_info["group"].unique().tolist() if "group" in sample_info.columns else ["Unknown"]
    gcolors = _dynamic_group_colors(present_groups)
    for s in samples:
        vals = norm_expr[s].dropna().values
        if len(vals) < 10:
            continue
        grp = sample_info.loc[s, "group"] if s in sample_info.index else "Unknown"
        color = gcolors.get(grp, MUTED)
        kde = gaussian_kde(vals, bw_method=0.3)
        x = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(x, kde(x), color=color, linewidth=1.2, alpha=0.6)

    handles = [mpatches.Patch(color=gcolors[g], label=g, alpha=0.7) for g in present_groups]
    ax.legend(handles=handles, framealpha=0.15, fontsize=8)
    ax.set_xlabel("log₂ Expression")
    ax.set_ylabel("Density")
    ax.set_title("Per-Sample Expression Density")
    fig.tight_layout()
    return _to_b64(fig)


# ──────────────────────────────────────────────────────────────────
# 10. DENDROGRAM PLOT
# ──────────────────────────────────────────────────────────────────

def plot_dendrogram(cluster_result: dict,
                    sample_info: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(max(10, len(cluster_result["samples"]) * 0.4), 5))

    link   = cluster_result["linkage_matrix"]
    labels = cluster_result["samples"]

    all_grps = list(dict.fromkeys(
        sample_info.loc[s, "group"] if s in sample_info.index else "Unknown"
        for s in labels
    ))
    gcolors = _dynamic_group_colors(all_grps)

    ddata = dendrogram(
        link,
        labels=[s[-6:] for s in labels],
        ax=ax,
        color_threshold=0,
        above_threshold_color=SUBTLE,
        leaf_rotation=60,
        leaf_font_size=7,
    )

    # Color leaf labels
    xlbls = ax.get_xmajorticklabels()
    for lbl, orig_s in zip(xlbls, [labels[i] for i in ddata["leaves"]]):
        grp = sample_info.loc[orig_s, "group"] if orig_s in sample_info.index else "Unknown"
        lbl.set_color(gcolors.get(grp, MUTED))

    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_ylabel(f"Distance ({cluster_result['metric']})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _to_b64(fig)
