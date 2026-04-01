"""
Post-pyBMDS Downstream Analysis Modules.

Functions for:
    - Quality control and filtering of BMD results
    - BMD distribution analysis (histograms, summary stats)
    - Pathway-level BMD analysis (KEGG, Reactome, GO mapping)
    - Transcriptomic Point of Departure (tPOD) derivation
    - Visualisation utilities

Designed to work with output from pybmds_modules.py
"""

import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Quality Control & Filtering
# =============================================================================

def qc_filter_results(
    bmd_results: pd.DataFrame,
    min_gof_pvalue: float = 0.1,
    max_bmd: Optional[float] = None,
    max_bmdu_bmdl_ratio: Optional[float] = None,
    require_positive_bmd: bool = True,
    require_finite_bmdl: bool = True,
    bmd_col: str = "bmd",
    bmdl_col: str = "bmdl",
    bmdu_col: str = "bmdu",
    pvalue_col: str = "pvalue",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply standard QC filters to BMD results.

    Follows EPA/toxicogenomics best practices for filtering unreliable
    BMD estimates before downstream analysis.

    Args:
        bmd_results: DataFrame from fit_all_genes() or
                     fit_all_genes_full_output().
        min_gof_pvalue: Minimum goodness-of-fit p-value (default: 0.1,
                        per EPA guidance).
        max_bmd: Maximum allowed BMD value. Genes with BMD above this
                 are excluded (useful for removing unrealistic fits that
                 extrapolate far beyond tested doses).
        max_bmdu_bmdl_ratio: Maximum allowed BMDU/BMDL ratio. A very
                             wide confidence interval suggests the BMD
                             is poorly constrained.
        require_positive_bmd: Exclude genes with BMD <= 0 (default: True).
        require_finite_bmdl: Exclude genes with missing/infinite BMDL
                             (default: True).
        bmd_col: Column name for BMD values.
        bmdl_col: Column name for BMDL values.
        bmdu_col: Column name for BMDU values.
        pvalue_col: Column name for p-values.
        verbose: Print filter summary.

    Returns:
        Filtered DataFrame with a 'qc_pass' column added.
    """
    df = bmd_results.copy()
    n_start = len(df)

    # Track which filter each gene fails
    df["qc_pass"] = True
    df["qc_fail_reason"] = ""

    # 1. Must have converged
    if "converged" in df.columns:
        mask = df["converged"] != True
        df.loc[mask, "qc_pass"] = False
        df.loc[mask, "qc_fail_reason"] += "no_convergence;"

    # 2. Goodness-of-fit p-value
    if pvalue_col in df.columns and min_gof_pvalue is not None:
        mask = df[pvalue_col] < min_gof_pvalue
        df.loc[mask, "qc_pass"] = False
        df.loc[mask, "qc_fail_reason"] += "low_gof_pvalue;"

    # 3. Positive BMD
    if require_positive_bmd and bmd_col in df.columns:
        mask = (df[bmd_col] <= 0) | df[bmd_col].isna()
        df.loc[mask, "qc_pass"] = False
        df.loc[mask, "qc_fail_reason"] += "non_positive_bmd;"

    # 4. Finite BMDL
    if require_finite_bmdl and bmdl_col in df.columns:
        mask = df[bmdl_col].isna() | np.isinf(df[bmdl_col])
        df.loc[mask, "qc_pass"] = False
        df.loc[mask, "qc_fail_reason"] += "missing_bmdl;"

    # 5. Maximum BMD
    if max_bmd is not None and bmd_col in df.columns:
        mask = df[bmd_col] > max_bmd
        df.loc[mask, "qc_pass"] = False
        df.loc[mask, "qc_fail_reason"] += "bmd_too_high;"

    # 6. BMDU/BMDL ratio (confidence interval width)
    if (
        max_bmdu_bmdl_ratio is not None
        and bmdl_col in df.columns
        and bmdu_col in df.columns
    ):
        ratio = df[bmdu_col] / df[bmdl_col].replace(0, np.nan)
        mask = ratio > max_bmdu_bmdl_ratio
        df.loc[mask, "qc_pass"] = False
        df.loc[mask, "qc_fail_reason"] += "wide_ci;"

    n_pass = df["qc_pass"].sum()
    n_fail = n_start - n_pass

    if verbose:
        print(f"QC filtering: {n_pass}/{n_start} passed ({n_fail} removed)")
        if n_fail > 0:
            reasons = (
                df.loc[~df["qc_pass"], "qc_fail_reason"]
                .str.strip(";")
                .str.split(";")
                .explode()
                .value_counts()
            )
            for reason, count in reasons.items():
                print(f"  - {reason}: {count}")

    return df


def get_qc_passed(qc_results: pd.DataFrame) -> pd.DataFrame:
    """Return only the genes that passed QC."""
    return qc_results[qc_results["qc_pass"] == True].copy()


# =============================================================================
# BMD Distribution Analysis
# =============================================================================

def bmd_distribution_stats(
    bmd_results: pd.DataFrame,
    bmd_col: str = "bmd",
    bmdl_col: str = "bmdl",
) -> Dict[str, Any]:
    """
    Compute summary statistics for the BMD distribution.

    These statistics describe the overall sensitivity landscape:
    which doses start to perturb gene expression.

    Args:
        bmd_results: Filtered DataFrame (post-QC recommended).
        bmd_col: Column name for BMD values.
        bmdl_col: Column name for BMDL values.

    Returns:
        Dictionary with distribution statistics.
    """
    bmd_vals = bmd_results[bmd_col].dropna()
    bmdl_vals = bmd_results[bmdl_col].dropna() if bmdl_col in bmd_results.columns else pd.Series(dtype=float)

    stats: Dict[str, Any] = {
        "n_genes": len(bmd_vals),
        "bmd_mean": bmd_vals.mean(),
        "bmd_median": bmd_vals.median(),
        "bmd_std": bmd_vals.std(),
        "bmd_min": bmd_vals.min(),
        "bmd_max": bmd_vals.max(),
        "bmd_q05": bmd_vals.quantile(0.05),
        "bmd_q10": bmd_vals.quantile(0.10),
        "bmd_q25": bmd_vals.quantile(0.25),
        "bmd_q75": bmd_vals.quantile(0.75),
        "bmd_q90": bmd_vals.quantile(0.90),
        "bmd_q95": bmd_vals.quantile(0.95),
        "bmd_geometric_mean": np.exp(np.log(bmd_vals[bmd_vals > 0]).mean()) if (bmd_vals > 0).any() else np.nan,
    }

    if len(bmdl_vals) > 0:
        stats.update({
            "bmdl_median": bmdl_vals.median(),
            "bmdl_q05": bmdl_vals.quantile(0.05),
            "bmdl_q10": bmdl_vals.quantile(0.10),
        })

    return stats


def rank_genes_by_sensitivity(
    bmd_results: pd.DataFrame,
    bmd_col: str = "bmd",
    gene_col: str = "gene",
    ascending: bool = True,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rank genes by BMD (most sensitive first).

    The most sensitive genes (lowest BMD) are the earliest responders
    to chemical exposure and are typically of greatest interest.

    Args:
        bmd_results: Filtered DataFrame.
        bmd_col: Column name for BMD values.
        gene_col: Column name for gene identifiers.
        ascending: If True (default), most sensitive (lowest BMD) first.
        top_n: Return only the top N most sensitive genes.

    Returns:
        DataFrame sorted by BMD with a 'sensitivity_rank' column.
    """
    ranked = bmd_results.dropna(subset=[bmd_col]).copy()
    ranked = ranked.sort_values(bmd_col, ascending=ascending).reset_index(drop=True)
    ranked["sensitivity_rank"] = range(1, len(ranked) + 1)

    if top_n is not None:
        ranked = ranked.head(top_n)

    return ranked


# =============================================================================
# Pathway-Level BMD Analysis
# =============================================================================

def load_gene_pathway_mapping(
    pathway_file: str,
    gene_col: str = "gene",
    pathway_col: str = "pathway",
    pathway_name_col: Optional[str] = "pathway_name",
    sep: str = "\t",
) -> pd.DataFrame:
    """
    Load a gene-to-pathway mapping file.

    The mapping file should have at least two columns: one for gene
    identifiers and one for pathway identifiers. Optionally a third
    column with human-readable pathway names.

    Common sources:
        - MSigDB GMT files (converted to tabular)
        - KEGG pathway downloads
        - Reactome gene-pathway associations
        - GO annotations

    Args:
        pathway_file: Path to the mapping file (TSV/CSV).
        gene_col: Column name for gene identifiers.
        pathway_col: Column name for pathway identifiers.
        pathway_name_col: Column name for pathway names (optional).
        sep: Delimiter (default: tab).

    Returns:
        DataFrame with gene-pathway associations.
    """
    mapping = pd.read_csv(pathway_file, sep=sep)

    required = {gene_col, pathway_col}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(mapping.columns)}"
        )

    return mapping


def create_pathway_mapping_from_gmt(gmt_file: str) -> pd.DataFrame:
    """
    Parse an MSigDB GMT file into a gene-pathway mapping DataFrame.

    GMT format: each line is
        pathway_name<tab>description<tab>gene1<tab>gene2<tab>...

    Args:
        gmt_file: Path to the GMT file.

    Returns:
        DataFrame with columns [pathway, pathway_name, gene].
    """
    records = []
    with open(gmt_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pathway_name = parts[0]
            # parts[1] is the description/URL, skip it
            genes = parts[2:]
            for gene in genes:
                if gene.strip():
                    records.append({
                        "pathway": pathway_name,
                        "pathway_name": pathway_name,
                        "gene": gene.strip(),
                    })

    return pd.DataFrame(records)


def compute_pathway_bmd(
    bmd_results: pd.DataFrame,
    gene_pathway_mapping: pd.DataFrame,
    bmd_col: str = "bmd",
    bmdl_col: str = "bmdl",
    gene_col: str = "gene",
    pathway_col: str = "pathway",
    pathway_name_col: Optional[str] = "pathway_name",
    min_genes_per_pathway: int = 3,
    aggregation: str = "median",
) -> pd.DataFrame:
    """
    Compute pathway-level BMD values.

    For each pathway, aggregates the BMDs of its constituent genes
    to produce a single pathway-level BMD estimate.

    The median is the standard choice: it is robust to outliers
    and represents the "typical" gene response within the pathway.

    Args:
        bmd_results: Filtered DataFrame with per-gene BMD results.
        gene_pathway_mapping: DataFrame mapping genes to pathways.
        bmd_col: Column name for BMD values.
        bmdl_col: Column name for BMDL values.
        gene_col: Column name for gene identifiers.
        pathway_col: Column name for pathway identifiers.
        pathway_name_col: Column name for pathway names.
        min_genes_per_pathway: Minimum number of genes with valid BMDs
                               required for a pathway to be included
                               (default: 3).
        aggregation: How to aggregate gene BMDs ("median", "mean",
                     "geometric_mean", "fifth_percentile").

    Returns:
        DataFrame with one row per pathway, containing:
            - pathway, pathway_name
            - pathway_bmd (aggregated BMD)
            - pathway_bmdl (aggregated BMDL)
            - n_genes_total (genes in pathway present in data)
            - n_genes_with_bmd (genes with valid BMD fits)
            - gene_list (semicolon-separated gene names)
            - bmd_values (semicolon-separated BMD values)
    """
    # Merge BMD results with pathway mapping
    valid_bmd = bmd_results.dropna(subset=[bmd_col])[[gene_col, bmd_col, bmdl_col]].copy()

    merged = gene_pathway_mapping.merge(
        valid_bmd, on=gene_col, how="inner"
    )

    if len(merged) == 0:
        print("Warning: No genes matched between BMD results and pathway mapping.")
        print(f"  BMD gene examples: {list(valid_bmd[gene_col].head(5))}")
        print(f"  Pathway gene examples: {list(gene_pathway_mapping[gene_col].head(5))}")
        return pd.DataFrame()

    # Aggregation function
    def _aggregate(values: pd.Series, method: str) -> float:
        if method == "median":
            return values.median()
        elif method == "mean":
            return values.mean()
        elif method == "geometric_mean":
            pos = values[values > 0]
            return np.exp(np.log(pos).mean()) if len(pos) > 0 else np.nan
        elif method == "fifth_percentile":
            return values.quantile(0.05)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    # Group by pathway
    pathway_results = []
    for pathway_id, group in merged.groupby(pathway_col):
        n_genes = len(group)
        if n_genes < min_genes_per_pathway:
            continue

        pw_bmd = _aggregate(group[bmd_col], aggregation)
        pw_bmdl = _aggregate(group[bmdl_col].dropna(), aggregation) if bmdl_col in group.columns else np.nan

        row = {
            "pathway": pathway_id,
            "pathway_bmd": pw_bmd,
            "pathway_bmdl": pw_bmdl,
            "n_genes_with_bmd": n_genes,
            "gene_list": ";".join(group[gene_col].tolist()),
            "bmd_values": ";".join(f"{v:.4g}" for v in group[bmd_col].tolist()),
        }

        # Add pathway name if available
        if pathway_name_col and pathway_name_col in group.columns:
            row["pathway_name"] = group[pathway_name_col].iloc[0]

        pathway_results.append(row)

    result = pd.DataFrame(pathway_results)

    if len(result) > 0:
        result = result.sort_values("pathway_bmd").reset_index(drop=True)
        result["pathway_rank"] = range(1, len(result) + 1)

    return result


# =============================================================================
# Transcriptomic Point of Departure (tPOD)
# =============================================================================

def compute_transcriptomic_pod(
    bmd_results: pd.DataFrame,
    pathway_bmd: Optional[pd.DataFrame] = None,
    method: str = "pathway_median",
    bmd_col: str = "bmd",
    bmdl_col: str = "bmdl",
    percentile: float = 10,
    n_sensitive_pathways: int = 1,
) -> Dict[str, Any]:
    """
    Derive the transcriptomic Point of Departure (tPOD).

    The tPOD represents the dose at which meaningful molecular
    perturbation begins. It serves as the starting point for risk
    assessment, analogous to a NOAEL or apical BMD.

    Several approaches are used in the literature:

    - "pathway_median": BMDL of the most sensitive pathway
      (lowest median-BMD pathway). Requires pathway_bmd input.
    - "gene_percentile": Nth percentile of the gene-level BMD
      distribution (e.g., 10th percentile).
    - "first_mode": First mode of the gene-level BMD distribution,
      representing the dose at which the first wave of gene
      perturbation occurs.

    Args:
        bmd_results: Filtered per-gene BMD results.
        pathway_bmd: Pathway-level BMD results from
                     compute_pathway_bmd(). Required for
                     "pathway_median" method.
        method: POD derivation method (see above).
        bmd_col: Column name for BMD values.
        bmdl_col: Column name for BMDL values.
        percentile: Percentile to use for "gene_percentile" method.
        n_sensitive_pathways: Number of most sensitive pathways to
                             consider for "pathway_median" method.

    Returns:
        Dictionary with:
            - tpod: The transcriptomic POD value
            - tpod_bmdl: Lower bound on the tPOD (when available)
            - method: Method used
            - details: Supporting information
    """
    result: Dict[str, Any] = {
        "tpod": np.nan,
        "tpod_bmdl": np.nan,
        "method": method,
        "details": {},
    }

    bmd_vals = bmd_results[bmd_col].dropna()

    if method == "pathway_median":
        if pathway_bmd is None or len(pathway_bmd) == 0:
            raise ValueError(
                "pathway_bmd is required for 'pathway_median' method. "
                "Run compute_pathway_bmd() first."
            )

        top_pathways = pathway_bmd.nsmallest(n_sensitive_pathways, "pathway_bmd")
        most_sensitive = top_pathways.iloc[0]

        result["tpod"] = most_sensitive["pathway_bmd"]
        result["tpod_bmdl"] = most_sensitive.get("pathway_bmdl", np.nan)
        result["details"] = {
            "pathway": most_sensitive["pathway"],
            "pathway_name": most_sensitive.get("pathway_name", ""),
            "n_genes": most_sensitive["n_genes_with_bmd"],
            "top_pathways": top_pathways[
                ["pathway", "pathway_bmd", "pathway_bmdl", "n_genes_with_bmd"]
            ].to_dict("records"),
        }

    elif method == "gene_percentile":
        if len(bmd_vals) == 0:
            return result

        tpod = bmd_vals.quantile(percentile / 100)
        result["tpod"] = tpod
        result["details"] = {
            "percentile": percentile,
            "n_genes": len(bmd_vals),
            "genes_below_tpod": int((bmd_vals <= tpod).sum()),
        }

        # Approximate BMDL-based tPOD
        if bmdl_col in bmd_results.columns:
            bmdl_vals = bmd_results[bmdl_col].dropna()
            if len(bmdl_vals) > 0:
                result["tpod_bmdl"] = bmdl_vals.quantile(percentile / 100)

    elif method == "first_mode":
        if len(bmd_vals) < 10:
            result["details"] = {"error": "Too few genes for mode estimation"}
            return result

        # Use kernel density estimation to find first mode
        try:
            from scipy.stats import gaussian_kde

            log_bmd = np.log10(bmd_vals[bmd_vals > 0])
            kde = gaussian_kde(log_bmd)

            x_grid = np.linspace(log_bmd.min(), log_bmd.max(), 500)
            density = kde(x_grid)

            # Find peaks (local maxima)
            peaks = []
            for i in range(1, len(density) - 1):
                if density[i] > density[i - 1] and density[i] > density[i + 1]:
                    peaks.append((x_grid[i], density[i]))

            if peaks:
                # First mode = leftmost peak (lowest BMD concentration)
                first_mode_log = peaks[0][0]
                result["tpod"] = 10 ** first_mode_log
                result["details"] = {
                    "n_modes_found": len(peaks),
                    "all_modes_log10": [p[0] for p in peaks],
                    "all_modes_bmd": [10 ** p[0] for p in peaks],
                }
            else:
                result["details"] = {"error": "No modes found in distribution"}

        except ImportError:
            result["details"] = {
                "error": "scipy required for first_mode method. "
                         "Install with: pip install scipy"
            }

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from: pathway_median, gene_percentile, first_mode"
        )

    return result


# =============================================================================
# Reporting Utilities
# =============================================================================

def generate_summary_report(
    bmd_results: pd.DataFrame,
    qc_results: Optional[pd.DataFrame] = None,
    pathway_bmd: Optional[pd.DataFrame] = None,
    tpod: Optional[Dict[str, Any]] = None,
    bmd_col: str = "bmd",
    bmdl_col: str = "bmdl",
) -> str:
    """
    Generate a text summary report of the analysis.

    Args:
        bmd_results: Raw BMD results.
        qc_results: QC-filtered results (optional).
        pathway_bmd: Pathway-level BMD results (optional).
        tpod: Transcriptomic POD result (optional).
        bmd_col: Column name for BMD.
        bmdl_col: Column name for BMDL.

    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("TOXICOGENOMIC BMD ANALYSIS REPORT")
    lines.append("=" * 60)

    # Overview
    lines.append("\n1. BMD FITTING OVERVIEW")
    lines.append("-" * 40)
    n_total = len(bmd_results)
    n_converged = bmd_results["converged"].sum() if "converged" in bmd_results.columns else "N/A"
    lines.append(f"  Total genes analysed:  {n_total}")
    lines.append(f"  Models converged:      {n_converged}")
    if isinstance(n_converged, (int, np.integer)):
        lines.append(f"  Convergence rate:      {100 * n_converged / n_total:.1f}%")

    # Model distribution
    model_col = "model" if "model" in bmd_results.columns else "model_name"
    if model_col in bmd_results.columns:
        conv = bmd_results[bmd_results.get("converged", True) == True]
        if len(conv) > 0:
            lines.append(f"\n  Best model distribution:")
            for model, count in conv[model_col].value_counts().items():
                lines.append(f"    {model}: {count} ({100 * count / len(conv):.1f}%)")

    # QC results
    if qc_results is not None:
        lines.append("\n2. QUALITY CONTROL")
        lines.append("-" * 40)
        n_pass = qc_results["qc_pass"].sum()
        n_fail = len(qc_results) - n_pass
        lines.append(f"  Passed QC:   {n_pass}")
        lines.append(f"  Failed QC:   {n_fail}")
        lines.append(f"  Pass rate:   {100 * n_pass / len(qc_results):.1f}%")

    # BMD distribution
    passed = qc_results[qc_results["qc_pass"] == True] if qc_results is not None else bmd_results
    bmd_vals = passed[bmd_col].dropna()
    if len(bmd_vals) > 0:
        lines.append("\n3. BMD DISTRIBUTION (post-QC)")
        lines.append("-" * 40)
        lines.append(f"  N genes:     {len(bmd_vals)}")
        lines.append(f"  Mean:        {bmd_vals.mean():.4g}")
        lines.append(f"  Median:      {bmd_vals.median():.4g}")
        lines.append(f"  Std Dev:     {bmd_vals.std():.4g}")
        lines.append(f"  Min:         {bmd_vals.min():.4g}")
        lines.append(f"  Max:         {bmd_vals.max():.4g}")
        lines.append(f"  5th %ile:    {bmd_vals.quantile(0.05):.4g}")
        lines.append(f"  10th %ile:   {bmd_vals.quantile(0.10):.4g}")
        lines.append(f"  25th %ile:   {bmd_vals.quantile(0.25):.4g}")
        lines.append(f"  75th %ile:   {bmd_vals.quantile(0.75):.4g}")

    # Pathway results
    if pathway_bmd is not None and len(pathway_bmd) > 0:
        lines.append(f"\n4. PATHWAY-LEVEL BMD")
        lines.append("-" * 40)
        lines.append(f"  Pathways analysed:  {len(pathway_bmd)}")
        lines.append(f"\n  Top 10 most sensitive pathways:")
        top10 = pathway_bmd.head(10)
        for _, row in top10.iterrows():
            name = row.get("pathway_name", row["pathway"])
            lines.append(
                f"    {row['pathway_rank']:3d}. {name[:45]:<45s}  "
                f"BMD={row['pathway_bmd']:.4g}  "
                f"(n={row['n_genes_with_bmd']})"
            )

    # Transcriptomic POD
    if tpod is not None and not np.isnan(tpod.get("tpod", np.nan)):
        lines.append(f"\n5. TRANSCRIPTOMIC POINT OF DEPARTURE")
        lines.append("-" * 40)
        lines.append(f"  Method:   {tpod['method']}")
        lines.append(f"  tPOD:     {tpod['tpod']:.4g}")
        if not np.isnan(tpod.get("tpod_bmdl", np.nan)):
            lines.append(f"  tPOD-L:   {tpod['tpod_bmdl']:.4g}")

        if "pathway" in tpod.get("details", {}):
            lines.append(f"  Pathway:  {tpod['details'].get('pathway_name', tpod['details']['pathway'])}")
            lines.append(f"  N genes:  {tpod['details']['n_genes']}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def export_all_results(
    output_dir: str,
    bmd_results: pd.DataFrame,
    qc_results: Optional[pd.DataFrame] = None,
    pathway_bmd: Optional[pd.DataFrame] = None,
    tpod: Optional[Dict[str, Any]] = None,
    ranked_genes: Optional[pd.DataFrame] = None,
    prefix: str = "bmd_analysis",
) -> List[str]:
    """
    Export all analysis results to files in a single directory.

    Creates:
        - {prefix}_gene_results.csv: Per-gene BMD results
        - {prefix}_qc_results.csv: QC-annotated results (if provided)
        - {prefix}_pathway_bmd.csv: Pathway-level BMDs (if provided)
        - {prefix}_ranked_genes.csv: Sensitivity-ranked genes (if provided)
        - {prefix}_report.txt: Text summary report

    Args:
        output_dir: Directory to save files in.
        bmd_results: Per-gene BMD results.
        qc_results: QC-filtered results.
        pathway_bmd: Pathway-level BMD results.
        tpod: Transcriptomic POD result.
        ranked_genes: Sensitivity-ranked genes.
        prefix: Filename prefix.

    Returns:
        List of file paths created.
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # Gene results
    path = os.path.join(output_dir, f"{prefix}_gene_results.csv")
    bmd_results.to_csv(path, index=False)
    created_files.append(path)

    # QC results
    if qc_results is not None:
        path = os.path.join(output_dir, f"{prefix}_qc_results.csv")
        qc_results.to_csv(path, index=False)
        created_files.append(path)

    # Pathway BMD
    if pathway_bmd is not None:
        path = os.path.join(output_dir, f"{prefix}_pathway_bmd.csv")
        # Drop the semicolon-delimited columns for clean CSV
        clean = pathway_bmd.drop(
            columns=["gene_list", "bmd_values"], errors="ignore"
        )
        clean.to_csv(path, index=False)
        created_files.append(path)

    # Ranked genes
    if ranked_genes is not None:
        path = os.path.join(output_dir, f"{prefix}_ranked_genes.csv")
        ranked_genes.to_csv(path, index=False)
        created_files.append(path)

    # Text report
    report = generate_summary_report(
        bmd_results, qc_results, pathway_bmd, tpod
    )
    path = os.path.join(output_dir, f"{prefix}_report.txt")
    with open(path, "w") as f:
        f.write(report)
    created_files.append(path)
    print(report)

    return created_files