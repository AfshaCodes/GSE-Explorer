"""
app.py — GSE Explorer Flask Application
Full backend: GEO fetch → preprocess → DEG → PCA → enrichment → plots
"""

import io
import json
import logging
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (Flask, jsonify, render_template, request,
                   send_file, Response, abort)

# Analysis modules
from analysis.fetcher       import fetch_gse, clear_cache, list_cached
from analysis.preprocessor  import preprocess, split_by_group, split_by_all_groups
from analysis.deg_analysis   import run_deg_analysis, run_multigroup_deg, deg_summary, volcano_data
from analysis.pca_analysis   import run_pca, run_hierarchical_clustering, compute_gene_correlation, compute_sample_correlation
from analysis.enrichment     import run_enrichment
from analysis import visualizer as viz

# ── App setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gse_app")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"]  = False
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024  # 128 MB

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory session store (one dataset at a time; extend with Redis for prod)
SESSION: dict = {}


# ════════════════════════════════════════════════════════════════
# HELPER UTILS
# ════════════════════════════════════════════════════════════════

def _safe_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _df_to_records(df: pd.DataFrame, max_rows: int = 500) -> list:
    """Convert DataFrame to JSON-safe list of dicts."""
    sub = df.head(max_rows).copy()
    for col in sub.select_dtypes(include=[np.floating]).columns:
        sub[col] = sub[col].apply(lambda x: round(float(x), 6) if pd.notna(x) else None)
    return sub.to_dict(orient="records")


def _error(msg: str, code: int = 400):
    logger.error(msg)
    return jsonify({"error": msg}), code


# ════════════════════════════════════════════════════════════════
# PAGES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    cached = list_cached()
    return render_template("index.html", cached_datasets=cached)


# ════════════════════════════════════════════════════════════════
# API: ANALYZE — main pipeline
# ════════════════════════════════════════════════════════════════

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    POST /api/analyze
    Body: { "gse_id": "GSE15471", "max_samples": 40,
            "normalization": "quantile", "deg_method": "welch",
            "fc_threshold": 1.0, "pval_threshold": 0.05 }
    """
    body = request.get_json(silent=True) or {}
    gse_id        = str(body.get("gse_id", "")).strip().upper()
    max_samples   = int(body.get("max_samples",   40))
    normalization = str(body.get("normalization", "quantile"))
    deg_method    = str(body.get("deg_method",    "welch"))
    fc_threshold  = float(body.get("fc_threshold",  1.0))
    pval_threshold = float(body.get("pval_threshold", 0.05))

    if not gse_id:
        return _error("gse_id is required")
    if not gse_id.startswith("GSE"):
        gse_id = "GSE" + gse_id

    logger.info(f"=== Starting analysis: {gse_id} ===")

    try:
        # ── Step 1: Fetch ──────────────────────────────────────
        logger.info("Step 1/5: Fetching data...")
        geo_data = fetch_gse(gse_id, max_samples=max_samples)
        expr_raw    = geo_data["expression"]
        sample_info = geo_data["sample_info"]
        metadata    = geo_data["metadata"]

        # ── Step 2: Preprocess ─────────────────────────────────
        logger.info("Step 2/5: Preprocessing...")
        prep = preprocess(
            expr_raw, sample_info,
            log2_transform=True,
            normalize=normalization,
            filter_low=True,
        )
        norm_expr   = prep["norm_expr"]

        # Determine group layout
        group_expr = split_by_all_groups(norm_expr, sample_info)
        all_groups = list(group_expr.keys())
        n_groups   = len(all_groups)
        is_ts      = all(str(g).startswith("T_") for g in all_groups)

        logger.info(f"Groups detected ({n_groups}): {all_groups}"
                    + (" [TIME-SERIES]" if is_ts else ""))

        # Legacy binary split for backward-compat helpers
        ctrl_expr, treat_expr = split_by_group(norm_expr, sample_info)

        # ── Step 3: DEG analysis ───────────────────────────────
        logger.info("Step 3/5: DEG analysis...")
        if n_groups == 2:
            # Simple binary case — run standard pipeline
            deg_df = run_deg_analysis(
                norm_expr, ctrl_expr, treat_expr,
                method=deg_method,
                fc_threshold=fc_threshold,
                pval_threshold=pval_threshold,
            )
            multigroup_result = None
            summary = deg_summary(deg_df, fc_threshold, pval_threshold)
        else:
            # Multi-class or time-series
            multigroup_result = run_multigroup_deg(
                norm_expr, group_expr,
                method=deg_method,
                fc_threshold=fc_threshold,
                pval_threshold=pval_threshold,
            )
            deg_df  = multigroup_result["primary"]
            summary = {
                "n_total":       len(deg_df),
                "n_significant": int(deg_df["is_sig_deg"].sum()),
                "n_groups":      n_groups,
                "groups":        all_groups,
                "is_timeseries": is_ts,
                "strategy":      multigroup_result["strategy"],
                "comparisons":   list(multigroup_result["pairwise"].keys()),
            }

        # ── Step 4: PCA + Clustering ───────────────────────────
        logger.info("Step 4/5: PCA + Clustering...")
        pca_result   = run_pca(norm_expr, sample_info)
        clust_result = run_hierarchical_clustering(norm_expr, sample_info)
        gene_corr    = compute_gene_correlation(norm_expr, top_n=40)
        samp_corr    = compute_sample_correlation(norm_expr)

        # ── Step 5: Enrichment ─────────────────────────────────
        logger.info("Step 5/5: Pathway enrichment...")
        enrich_df = run_enrichment(deg_df, fc_threshold, pval_threshold)

        # ── Store in session ───────────────────────────────────
        SESSION["gse_id"]             = gse_id
        SESSION["geo_data"]           = geo_data
        SESSION["prep"]               = prep
        SESSION["norm_expr"]          = norm_expr
        SESSION["sample_info"]        = sample_info
        SESSION["deg_df"]             = deg_df
        SESSION["multigroup_result"]  = multigroup_result
        SESSION["group_expr"]         = group_expr
        SESSION["all_groups"]         = all_groups
        SESSION["pca_result"]         = pca_result
        SESSION["clust_result"] = clust_result
        SESSION["gene_corr"]    = gene_corr
        SESSION["samp_corr"]    = samp_corr
        SESSION["enrich_df"]    = enrich_df
        SESSION["params"] = {
            "fc_threshold":  fc_threshold,
            "pval_threshold": pval_threshold,
            "deg_method":    deg_method,
            "normalization": normalization,
        }

        # ── Build response ─────────────────────────────────────
        group_counts = {grp: int(len(df.columns)) for grp, df in group_expr.items()}
        n_ctrl  = group_counts.get("Control", group_counts.get(all_groups[0], 0))
        n_treat = group_counts.get("Treatment", 0) or (
            sum(group_counts.values()) - n_ctrl if len(all_groups) == 2 else 0
        )

        response = {
            "gse_id":   gse_id,
            "source":   geo_data.get("source", "unknown"),
            "warning":  geo_data.get("warning"),
            "metadata": {
                "title":      metadata.get("title", gse_id),
                "summary":    metadata.get("summary", "")[:400],
                "organism":   metadata.get("organism", "Unknown"),
                "platform":   str(metadata.get("platform", [""])[0]) if isinstance(metadata.get("platform"), list) else str(metadata.get("platform", "")),
                "n_samples":  int(norm_expr.shape[1]),
                "pubmed_id":  metadata.get("pubmed_id"),
                "type":       metadata.get("type", ""),
            },
            "qc": {
                "n_genes_raw":      prep["n_genes_raw"],
                "n_genes_filtered": prep["n_genes_filtered"],
                "n_samples":        prep["n_samples"],
                "n_control":        n_ctrl,
                "n_treatment":      n_treat,
                "n_groups":         n_groups,
                "groups":           all_groups,
                "group_counts":     group_counts,
                "is_timeseries":    is_ts,
                "outlier_samples":  prep["outlier_samples"],
                "normalization":    prep["normalization"],
                "global":           prep["qc_metrics"]["global"],
                "inter_sample_corr_mean": prep["qc_metrics"]["global"]["inter_sample_corr_mean"],
            },
            "deg_summary": summary,
            "pca": {
                "variance_explained": pca_result["variance_explained"],
                "cumulative_var":     pca_result["cumulative_var"],
                "n_components":       pca_result["n_components"],
                "n_genes_used":       pca_result["n_genes_used"],
                "top_loadings":       {
                    pc: {
                        "positive": list(v["positive"].keys())[:5],
                        "negative": list(v["negative"].keys())[:5],
                    }
                    for pc, v in pca_result["top_loadings"].items()
                },
            },
            "enrichment_summary": {
                "n_tested":       int(len(enrich_df)),
                "n_significant":  int(enrich_df["significant"].sum()) if not enrich_df.empty else 0,
                "top_pathway":    enrich_df.iloc[0]["pathway"] if not enrich_df.empty else "—",
                "top_pval":       float(enrich_df.iloc[0]["pval"]) if not enrich_df.empty else 1.0,
            },
        }

        logger.info(f"=== Analysis complete: {gse_id} ===")
        return jsonify(response), 200

    except Exception as e:
        logger.error(traceback.format_exc())
        return _error(f"Analysis failed: {str(e)}", 500)


# ════════════════════════════════════════════════════════════════
# API: DATA ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/api/groups")
def api_groups():
    """Return available group labels and comparisons for multi-class datasets."""
    if "all_groups" not in SESSION:
        return _error("No analysis loaded.", 404)
    mg = SESSION.get("multigroup_result")
    return jsonify({
        "groups":       SESSION["all_groups"],
        "n_groups":     len(SESSION["all_groups"]),
        "is_timeseries": mg["is_timeseries"] if mg else False,
        "strategy":     mg["strategy"] if mg else "pairwise",
        "comparisons":  list(mg["pairwise"].keys()) if mg else [],
        "has_anova":    mg is not None and mg.get("anova") is not None,
    })


@app.route("/api/deg_comparison")
def api_deg_comparison():
    """
    Return DEG table for a specific pairwise comparison or ANOVA.
    Query params: comparison=<label>  (use 'anova' for ANOVA results)
    """
    if "multigroup_result" not in SESSION or SESSION["multigroup_result"] is None:
        # Fall back to primary deg_df for binary datasets
        return api_deg_table()

    mg    = SESSION["multigroup_result"]
    label = request.args.get("comparison", "anova")

    if label == "anova" and mg.get("anova") is not None:
        deg = mg["anova"].copy()
    elif label in mg["pairwise"]:
        deg = mg["pairwise"][label].copy()
    else:
        return _error(f"Comparison '{label}' not found. Available: anova, "
                      + ", ".join(mg["pairwise"].keys()), 404)

    return jsonify({
        "comparison": label,
        "total":      len(deg),
        "n_sig":      int(deg["is_sig_deg"].sum()),
        "rows":       _df_to_records(deg),
    })


@app.route("/api/deg_table")
def api_deg_table():
    """Return DEG table as JSON (paginated)."""
    if "deg_df" not in SESSION:
        return _error("No analysis loaded. Run /api/analyze first.", 404)

    page      = int(request.args.get("page",  1))
    per_page  = int(request.args.get("per_page", 50))
    sort_col  = request.args.get("sort", "padj")
    sort_dir  = request.args.get("dir", "asc")
    query     = request.args.get("q", "").lower()
    filter_   = request.args.get("filter", "all")  # all | up | down | sig

    deg = SESSION["deg_df"].copy()

    # Filter
    if query:
        deg = deg[deg["gene"].str.lower().str.contains(query)]
    if filter_ == "up":
        deg = deg[deg["is_sig_deg"] & (deg["log2fc"] > 0)]
    elif filter_ == "down":
        deg = deg[deg["is_sig_deg"] & (deg["log2fc"] < 0)]
    elif filter_ == "sig":
        deg = deg[deg["is_sig_deg"]]

    # Sort
    asc = sort_dir == "asc"
    if sort_col in deg.columns:
        deg = deg.sort_values(sort_col, ascending=asc)

    total   = len(deg)
    offset  = (page - 1) * per_page
    page_df = deg.iloc[offset : offset + per_page]

    return jsonify({
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    (total + per_page - 1) // per_page,
        "rows":     _df_to_records(page_df),
    })


@app.route("/api/pca_data")
def api_pca_data():
    """Return PCA coordinates for all samples."""
    if "pca_result" not in SESSION:
        return _error("No analysis loaded.", 404)
    coord_df = SESSION["pca_result"]["coordinates"]
    return jsonify({
        "points": _df_to_records(coord_df.reset_index()),
        "variance_explained": SESSION["pca_result"]["variance_explained"],
    })


@app.route("/api/gene_corr")
def api_gene_corr():
    """Return gene correlation matrix."""
    if "gene_corr" not in SESSION:
        return _error("No analysis loaded.", 404)
    gc = SESSION["gene_corr"]
    cm = gc["corr_matrix"]
    return jsonify({
        "genes":      gc["genes"],
        "matrix":     cm.values.round(4).tolist(),
        "top_pairs":  gc["top_pairs"][:20],
    })


@app.route("/api/sample_corr")
def api_sample_corr():
    """Return sample correlation matrix."""
    if "samp_corr" not in SESSION:
        return _error("No analysis loaded.", 404)
    sc = SESSION["samp_corr"]
    return jsonify({
        "samples": sc.columns.tolist(),
        "matrix":  sc.values.round(4).tolist(),
    })


@app.route("/api/enrichment")
def api_enrichment():
    """Return pathway enrichment results."""
    if "enrich_df" not in SESSION:
        return _error("No analysis loaded.", 404)
    df = SESSION["enrich_df"]
    if df.empty:
        return jsonify({"pathways": [], "n_significant": 0})
    return jsonify({
        "pathways":      _df_to_records(df),
        "n_significant": int(df["significant"].sum()),
    })


@app.route("/api/qc_data")
def api_qc_data():
    """Return per-sample QC metrics."""
    if "prep" not in SESSION:
        return _error("No analysis loaded.", 404)
    return jsonify({
        "per_sample": SESSION["prep"]["qc_metrics"]["per_sample"],
        "global":     SESSION["prep"]["qc_metrics"]["global"],
    })


@app.route("/api/expression_sample")
def api_expression_sample():
    """Return expression values for a specific gene."""
    if "norm_expr" not in SESSION:
        return _error("No analysis loaded.", 404)
    gene = request.args.get("gene", "")
    norm_expr   = SESSION["norm_expr"]
    sample_info = SESSION["sample_info"]
    if gene not in norm_expr.index:
        return _error(f"Gene '{gene}' not found in dataset.", 404)
    vals = norm_expr.loc[gene]
    return jsonify({
        "gene":    gene,
        "samples": vals.index.tolist(),
        "values":  vals.values.round(4).tolist(),
        "groups":  [sample_info.loc[s,"group"] if s in sample_info.index else "Unknown" for s in vals.index],
    })


# ════════════════════════════════════════════════════════════════
# API: PLOT ENDPOINTS — return base64 PNGs
# ════════════════════════════════════════════════════════════════

def _require_session():
    if "norm_expr" not in SESSION:
        abort(400, "No analysis loaded. Run /api/analyze first.")


@app.route("/api/plot/boxplot")
def plot_boxplot():
    _require_session()
    try:
        img = viz.plot_sample_boxplot(SESSION["norm_expr"], SESSION["sample_info"])
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/volcano")
def plot_volcano():
    _require_session()
    try:
        params = SESSION.get("params", {})
        img = viz.plot_volcano(SESSION["deg_df"],
                               fc_thresh=params.get("fc_threshold", 1.0),
                               pval_thresh=params.get("pval_threshold", 0.05))
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/heatmap")
def plot_heatmap():
    _require_session()
    try:
        n_genes   = int(request.args.get("n_genes",   50))
        n_samples = int(request.args.get("n_samples", 30))
        img = viz.plot_heatmap(SESSION["norm_expr"], SESSION["sample_info"],
                               n_genes=n_genes, n_samples=n_samples)
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/pca")
def plot_pca():
    _require_session()
    try:
        img = viz.plot_pca(SESSION["pca_result"], SESSION["sample_info"])
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/violin")
def plot_violin():
    _require_session()
    try:
        img = viz.plot_violin(SESSION["norm_expr"], SESSION["sample_info"])
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/correlation")
def plot_correlation():
    _require_session()
    try:
        corr = SESSION["gene_corr"]["corr_matrix"]
        img  = viz.plot_correlation_heatmap(corr, title="Gene Correlation Matrix")
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/enrichment")
def plot_enrichment():
    _require_session()
    try:
        img = viz.plot_enrichment(SESSION["enrich_df"])
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/ma")
def plot_ma():
    _require_session()
    try:
        img = viz.plot_ma(SESSION["deg_df"])
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/density")
def plot_density():
    _require_session()
    try:
        img = viz.plot_density(SESSION["norm_expr"], SESSION["sample_info"])
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/plot/dendrogram")
def plot_dendrogram():
    _require_session()
    try:
        img = viz.plot_dendrogram(SESSION["clust_result"], SESSION["sample_info"])
        return jsonify({"image": img})
    except Exception as e:
        return _error(str(e), 500)


# ════════════════════════════════════════════════════════════════
# API: EXPORT ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/api/export/degs_csv")
def export_degs_csv():
    if "deg_df" not in SESSION:
        return _error("No analysis loaded.", 404)
    df  = SESSION["deg_df"]
    gse = SESSION.get("gse_id", "GSE")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={gse}_DEGs.csv"}
    )


@app.route("/api/export/degs_excel")
def export_degs_excel():
    if "deg_df" not in SESSION:
        return _error("No analysis loaded.", 404)
    gse  = SESSION.get("gse_id", "GSE")
    df   = SESSION["deg_df"]
    enr  = SESSION.get("enrich_df", pd.DataFrame())
    path = OUTPUT_DIR / f"{gse}_results.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="DEG Results", index=False)
        SESSION["norm_expr"].head(200).to_excel(writer, sheet_name="Expression Matrix")
        if not enr.empty:
            enr.to_excel(writer, sheet_name="Pathway Enrichment", index=False)
        SESSION["prep"]["qc_metrics"]["per_sample"] and \
            pd.DataFrame(SESSION["prep"]["qc_metrics"]["per_sample"]).to_excel(writer, sheet_name="QC Metrics", index=False)

    return send_file(path, as_attachment=True,
                     download_name=f"{gse}_results.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/api/export/expression_csv")
def export_expression_csv():
    if "norm_expr" not in SESSION:
        return _error("No analysis loaded.", 404)
    gse = SESSION.get("gse_id", "GSE")
    buf = io.StringIO()
    SESSION["norm_expr"].to_csv(buf)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={gse}_expression.csv"}
    )


@app.route("/api/export/enrichment_csv")
def export_enrichment_csv():
    if "enrich_df" not in SESSION:
        return _error("No analysis loaded.", 404)
    gse = SESSION.get("gse_id", "GSE")
    buf = io.StringIO()
    SESSION["enrich_df"].to_csv(buf, index=False)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={gse}_enrichment.csv"}
    )


# ════════════════════════════════════════════════════════════════
# API: UTILITY
# ════════════════════════════════════════════════════════════════

@app.route("/api/cache/clear", methods=["POST"])
def api_clear_cache():
    gse_id = request.get_json(silent=True, force=True).get("gse_id") if request.data else None
    clear_cache(gse_id)
    return jsonify({"cleared": gse_id or "all"})


@app.route("/api/cache/list")
def api_list_cache():
    return jsonify({"cached": list_cached()})


@app.route("/api/status")
def api_status():
    loaded = "gse_id" in SESSION
    return jsonify({
        "loaded":  loaded,
        "gse_id":  SESSION.get("gse_id"),
        "n_genes": int(SESSION["norm_expr"].shape[0]) if loaded and "norm_expr" in SESSION else 0,
        "n_samples": int(SESSION["norm_expr"].shape[1]) if loaded and "norm_expr" in SESSION else 0,
    })


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  GSE Explorer — Flask Backend")
    print("  http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
