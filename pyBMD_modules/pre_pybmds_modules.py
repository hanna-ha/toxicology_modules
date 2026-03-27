# pre_pybmds_modules.py

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


# =============================================================================
# Step 1: Load and subset
# =============================================================================

def load_and_subset(count_file, annotation_file, cell_line, compound_name, control_names):
    """Load data and subset to relevant samples (cell line + compound + controls)."""
    # counts = pd.read_csv(count_file, index_col=0)
    # annot = pd.read_csv(annotation_file)

    counts = count_file 
    annot = annotation_file
    # Flag controls
    annot["is_control"] = annot["compound"].isin(control_names)

    # Subset to this cell line: compound samples + control samples
    mask = (
        (annot["sample"] == cell_line)
        & ((annot["compound"] == compound_name) | annot["is_control"])
    )
    annot_sub = annot.loc[mask].copy()

    # if we have less than 3 replicates 
    if len(annot_sub) < 3:
        raise ValueError(
            f"Not enough samples for {cell_line} + {compound_name}: "
            f"found {len(annot_sub)}, need at least 3"
        )

    # Build condition column for DEG analysis
    # Clean dose column for controls
    annot_sub["dose"] = annot_sub["dose"].astype(str)
    annot_sub.loc[annot_sub["dose"].isin(["nan", "NaN", "", "None"]), "dose"] = "0"
    annot_sub["condition"] = np.where(annot_sub["is_control"], "control", annot_sub["dose"]) # what does it do here 

    # Subset count matrix to matching samples
    sample_names = annot_sub["sample_name"].tolist()
        
    # Validate all samples exist in count matrix
    missing = set(sample_names) - set(counts.columns)
    if missing:
        raise ValueError(f"Samples not found in count matrix: {missing}")

    counts_sub = counts[sample_names]

    print(f"Samples subset: {len(annot_sub)} ({annot_sub['is_control'].sum()} controls, "
          f"{(~annot_sub['is_control']).sum()} treatments)")

    return counts_sub, annot_sub



# =============================================================================
# Step 2 & 3: Normalize and DEG analysis
# =============================================================================

def run_normalization_and_deg(counts_sub, annot_sub):
    """
    Run pydeseq2: normalize counts and compute DEG results for each dose vs control.
    Returns normalized count matrix and DEG summary table.
    """
    # pydeseq2 expects samples as rows, genes as columns
    count_matrix = counts_sub.T
    count_matrix.index = annot_sub["sample_name"].values

    # Build metadata
    metadata = pd.DataFrame({
        "condition": annot_sub["condition"].values,
    }, index=annot_sub["sample_name"].values)

    # Run pydeseq2
    print("Running pydeseq2 normalization and DEG analysis...")
    dds = DeseqDataSet(
        counts=count_matrix,
        metadata=metadata,
        design="~condition",
        refit_cooks=True, # whether Cooks outlier should be refitted
        ref_level=["condition", "control"],
        # n_cpus = 8  
    )
    dds.deseq2()

    # ---- Normalized counts ----
    size_factors = pd.Series(dds.obs["size_factors"].values, index=dds.obs_names)
    norm_counts = counts_sub.div(size_factors[counts_sub.columns], axis=1)
    print(f"Normalization complete. Size factors range: "
          f"{size_factors.min():.3f} - {size_factors.max():.3f}")

    # ---- DEG results for each dose vs control ----
    conditions = [c for c in annot_sub["condition"].unique() if c != "control"]
    deg_frames = []

    for cond in conditions:
        stat = DeseqStats(dds, contrast=["condition", cond, "control"])
        stat.summary()
        res = stat.results_df.copy()
        res["gene"] = res.index
        res["dose"] = cond
        deg_frames.append(res)

    deg_all = pd.concat(deg_frames, ignore_index=True)

    deg_all = deg_all[["gene", "dose", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]]

    return norm_counts, deg_all


# =============================================================================
# Step 4: Filter significant genes
# =============================================================================

def filter_significant_genes(deg_summary, padj_threshold):
    """Filter genes that have padj < threshold at ANY dose."""
    # we might want to use different filtering here. 
    # however, filtering by padj < 0.05 in any sample dosage would be sufficient 
    significant = deg_summary.loc[deg_summary["padj"] < padj_threshold, "gene"].unique().tolist()
    print(f"{len(significant)} significant genes after filtering padj < 0.05")
    return significant


# =============================================================================
# Step 5: Identify numeric doses from annotation
# =============================================================================

def clean_sample_dose(annot_sub, control_names):
    """
    pyBMDs expect numerical values in dosage 
    Build a mapping of sample_name -> numeric_dose.
    Controls get dose = 0.0.
    Treatment samples get their dose from the annotation's dose column.
    """
    annot = annot_sub.copy()
    annot["is_control"] = annot["compound"].isin(control_names)

    # Convert dose to numeric, controls get 0.0
    annot["numeric_dose"] = pd.to_numeric(annot["dose"], errors="coerce")
    annot.loc[annot["is_control"], "numeric_dose"] = 0.0

    # Check for non-numeric doses in treatment samples
    dose_error = annot.loc[(~annot["is_control"]) & (annot["numeric_dose"].isna())]
    if len(dose_error) > 0:
        bad_labels = dose_error["dose"].unique().tolist()
        print(f"WARNING: Non-numeric dose values found: {bad_labels}")
        print("These samples will be excluded. Ensure dose column has numeric values.")
        annot = annot.dropna(subset=["numeric_dose"])

    return annot[["sample_name", "numeric_dose", "is_control"]]


# =============================================================================
# Step 6 & 7: Compute log2FC and summarize per dose group
# =============================================================================

def compute_log2fc_and_summarize(norm_counts, sample_doses, gene_list, pseudocount=1.0, min_sd = 0.001):
    """
    Compute per-sample log2FC relative to control mean, then summarize per dose group.

    For each gene:
        1. control_mean = mean of normalized counts across all control samples
        2. per-sample log2FC = log2((norm_count + pseudocount) / (control_mean + pseudocount))
        3. group by dose -> mean, sd, n of the log2FC values

    Returns long-format DataFrame: gene, dose, mean_log2fc, sd_log2fc, n
    """
    # Get control and all sample names
    control_samples = sample_doses.loc[sample_doses["is_control"], "sample_name"].tolist()

    print(f"hh check control_samples { control_samples}")

    all_samples = sample_doses["sample_name"].tolist()

    print(f"hh check all_Samples { all_samples}")

    # Subset normalized counts to relevant genes and samples
    norm_sub = norm_counts.loc[gene_list, all_samples]

    print(f"hh check norm_sub { norm_sub.columns}")

    # Control mean per gene
    control_mean = norm_sub[control_samples].mean(axis=1)

    # Per-sample log2FC
    log2fc = np.log2(
        (norm_sub + pseudocount).div((control_mean + pseudocount), axis=0)
    )

    # Build sample -> dose lookup
    dose_lookup = sample_doses.set_index("sample_name")["numeric_dose"]
    print(f"hh check dose_lookup { dose_lookup}")


    # Melt to long format: gene, sample_name, log2fc
    log2fc.index.name = "gene"
    log2fc_long = log2fc.reset_index().melt(
        id_vars="gene",
        var_name="sample_name",
        value_name="log2fc",
    )

    # Map sample to numeric dose
    log2fc_long["dose"] = log2fc_long["sample_name"].map(dose_lookup)

    # Summarize per gene + dose group
    summary = log2fc_long.groupby(["gene", "dose"])["log2fc"].agg(
        mean_log2fc="mean",
        sd_log2fc="std",
        n="count",
    ).reset_index()

    # Replace NaN sd (single replicate) with small value
    summary["sd_log2fc"] = summary["sd_log2fc"].fillna(min_sd)
    # Replace 0 sd with small value (pybmds needs non-zero stdev)
    # summary.loc[summary["sd_log2fc"] < min_sd, "sd_log2fc"] = min_sd
    summary.loc[summary["sd_log2fc"] == 0, "sd_log2fc"] = min_sd

    # Sort by gene, then dose
    summary = summary.sort_values(["gene", "dose"]).reset_index(drop=True)

    return summary
