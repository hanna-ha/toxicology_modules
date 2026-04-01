"""
Run the full pyBMDS pipeline for all cell line + compound combinations.

Reads a count matrix and annotation file, normalizes, computes log2FC,
and fits BMD models for every gene.

Usage:
    python run_pybmds.py \
        --counts counts.csv \
        --annotation annotation.csv \
        --controls "DMSO1,DMSO2" \
        --output-dir results/bmd
"""

import argparse
from pathlib import Path

import pandas as pd

from pre_pybmds_modules import (
    load_and_subset,
    run_normalization_and_deg,
    filter_significant_genes,
    clean_sample_dose,
    compute_log2fc_and_summarize,
)
from pybmds_modules import fit_all_genes_all_models, get_best_models


def run_one(counts, annot, cell_line, compound, control_names, output_dir,
            padj_threshold=0.05, pseudocount=1.0, n_jobs=1):
    """Run the full pipeline for a single cell_line + compound."""

    print(f"\n{'='*60}")
    print(f"  {cell_line} | {compound}")
    print(f"{'='*60}")

    # normalize and DEG
    counts_sub, annot_sub = load_and_subset(counts, annot, cell_line, compound, control_names)
    norm_counts, deg_summary = run_normalization_and_deg(counts_sub, annot_sub)

    # filter genes
    sig_genes = filter_significant_genes(deg_summary, padj_threshold)
    if len(sig_genes) == 0:
        print(f"  Skipping — no significant genes")
        return None

    # compute log2fc
    sample_doses = clean_sample_dose(annot_sub, control_names)
    bmd_input = compute_log2fc_and_summarize(norm_counts, sample_doses, sig_genes, pseudocount)

    # fit BMD models
    all_models = fit_all_genes_all_models(bmd_input, n_jobs=n_jobs)
    best = get_best_models(all_models)

    # save
    prefix = f"{cell_line}_{compound}"
    all_models.to_csv(output_dir / f"{prefix}_all_models.csv", index=False)
    best.to_csv(output_dir / f"{prefix}_best_models.csv", index=False)
    bmd_input.to_csv(output_dir / f"{prefix}_bmd_input.csv", index=False)

    n_converged = best["converged"].sum()
    print(f"  {n_converged}/{len(best)} genes converged")
    print(f"  Saved to {output_dir / prefix}_*.csv")

    return best


def main():
    parser = argparse.ArgumentParser(description="Run pyBMDS pipeline")
    parser.add_argument("--counts", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--output-dir", default="results/bmd")
    parser.add_argument("--padj-threshold", type=float, default=0.05)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel workers for BMD fitting")
    args = parser.parse_args()

    control_names = [c.strip() for c in args.controls.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data once
    counts = pd.read_csv(args.counts, index_col=0)
    annot = pd.read_csv(args.annotation)

    # find all cell_line + compound pairs
    treatments = annot[~annot["compound"].isin(control_names)]
    pairs = treatments[["sample", "compound"]].drop_duplicates().values.tolist()
    print(f"Found {len(pairs)} combinations")

    # run each one
    for cell_line, compound in pairs:
        run_one(counts, annot.copy(), cell_line, compound, control_names, output_dir,
                args.padj_threshold, args.pseudocount, args.n_jobs)

    print(f"\nAll done. Results in {output_dir}/")


if __name__ == "__main__":
    main()