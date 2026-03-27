import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def parse_args():
    if len(sys.argv) != 7:
        print("Usage: python normalize_and_filter.py <count_file> <annotation_file> "
              "<cell_line> <compound_name> <control_names> <output_dir>")
        sys.exit(1)

    return {
        "count_file": sys.argv[1],
        "annotation_file": sys.argv[2],
        "cell_line": sys.argv[3],
        "compound_name": sys.argv[4],
        "control_names": [c.strip() for c in sys.argv[5].split(",")],
        "output_dir": sys.argv[6],
    }


def load_and_subset(count_file, annotation_file, cell_line, compound_name, control_names):
    """Load data and subset to relevant samples (cell line + compound + controls)."""
    counts = pd.read_csv(count_file, index_col=0)
    annot = pd.read_csv(annotation_file)

    # Clean dose column
    annot["dose"] = annot["dose"].astype(str)
    annot.loc[annot["dose"].isin(["nan", "NaN", "", "None"]), "dose"] = "0"

    # Flag controls
    annot["is_control"] = annot["compound"].isin(control_names)

    # Subset to this cell line: compound samples + control samples
    mask = (
        (annot["sample"] == cell_line)
        & ((annot["compound"] == compound_name) | annot["is_control"])
    )
    annot_sub = annot.loc[mask].copy()

    if len(annot_sub) < 2:
        print(f"ERROR: Not enough samples for {cell_line} {compound_name}")
        sys.exit(1)

    # Build condition column
    annot_sub["condition"] = np.where(annot_sub["is_control"], "control", annot_sub["dose"])

    # Subset count matrix to matching samples
    sample_names = annot_sub["sample_name"].tolist()
    counts_sub = counts[sample_names]

    return counts_sub, annot_sub





def run_pydeseq2(counts_sub, annot_sub):
    """
    Run pydeseq2: normalization + DEG analysis for each dose vs control.
    Returns normalized counts and DEG summary table.
    """
    # pydeseq2 expects samples as rows, genes as columns
    count_matrix = counts_sub.T
    count_matrix.index = annot_sub["sample_name"].values

    # Build metadata
    metadata = pd.DataFrame({
        "condition": annot_sub["condition"].values,
    }, index=annot_sub["sample_name"].values)

    # Create DESeq2 dataset and fit
    dds = DeseqDataSet(
        counts=count_matrix,
        metadata=metadata,
        design="~condition",
        refit_cooks=True,
        ref_level=["condition", "control"],
    )
    dds.deseq2()

    size_factors = pd.Series(dds.obs["size_factors"], index=dds.obs_names)
    norm_counts = counts_sub.div(size_factors[counts_sub.columns], axis=1)
    

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

    # Standardize column names to match R output
    col_map = {
        "gene": "gene",
        "dose": "dose",
        "baseMean": "baseMean",
        "log2FoldChange": "log2FoldChange",
        "lfcSE": "lfcSE",
        "stat": "stat",
        "pvalue": "pvalue",
        "padj": "padj",
    }
    deg_all = deg_all.rename(columns=col_map)
    deg_all = deg_all[["gene", "dose", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]]

    return norm_counts, deg_all


def main():
    args = parse_args()

    # ---- Load and subset ----
    counts_sub, annot_sub = load_and_subset(
        args["count_file"],
        args["annotation_file"],
        args["cell_line"],
        args["compound_name"],
        args["control_names"],
    )


    # ---- Run normalization and DEG analysis ----
    norm_counts, deg_all = run_pydeseq2(counts_sub, annot_sub)

    # ---- Write outputs ----
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args['cell_line']}_{args['compound_name']}"

    # Output 1: Normalized count matrix
    norm_out = output_dir / f"{prefix}_normalized_counts.csv"
    norm_df = norm_counts.copy()
    norm_df.insert(0, "gene", norm_df.index)
    norm_df.to_csv(norm_out, index=False)
    print(f"Written: {norm_out}")

    # Output 2: DEG summary table
    deg_out = output_dir / f"{prefix}_deg_summary.csv"
    deg_all.to_csv(deg_out, index=False)
    print(f"Written: {deg_out}")


    print("Done.")


if __name__ == "__main__":
    main()