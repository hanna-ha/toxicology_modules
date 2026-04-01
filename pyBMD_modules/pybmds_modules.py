"""
pyBMDS model fitting module.

Fits dose-response models to gene-level summary data (from prepare_bmd_input.py)
and returns results for all models per gene.
"""

import os
import sys
import warnings
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pybmds
from pybmds import ContinuousRiskType




# =============================================================================
# Dataset Creation
# =============================================================================

def create_continuous_dataset(
    gene_data: pd.DataFrame,
    gene_name: str,
) -> pybmds.ContinuousDataset:
    """
    Create a pyBMDS ContinuousDataset for a single gene.

    Args:
        gene_data: DataFrame with columns [gene, dose, mean_log2fc, sd_log2fc, n].
                   If a 'gene' column is present the data is auto-filtered.
        gene_name: Gene identifier for labeling.

    Returns:
        pybmds.ContinuousDataset ready for model fitting.

    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    if "gene" in gene_data.columns:
        gene_data = gene_data[gene_data["gene"] == gene_name].copy()
        if len(gene_data) == 0:
            raise ValueError(f"Gene '{gene_name}' not found in data")

    required_cols = ["dose", "mean_log2fc", "sd_log2fc", "n"]
    missing = set(required_cols) - set(gene_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gene_data = gene_data.sort_values("dose").reset_index(drop=True)

    if gene_data["dose"].duplicated().any():
        n_genes = (
            gene_data["gene"].nunique()
            if "gene" in gene_data.columns
            else "unknown"
        )
        raise ValueError(
            f"Duplicate doses found. This usually means you passed the full "
            f"DataFrame instead of filtering to one gene first. "
            f"Found {len(gene_data)} rows, genes in data: {n_genes}. "
            f"Filter first: df[df['gene'] == '{gene_name}']"
        )

    if len(gene_data) < 3:
        raise ValueError(
            f"Need at least 3 dose groups for BMD modeling, got {len(gene_data)}. "
            f"With 4+ dose groups, more complex models (Hill) can be fitted."
        )

    if (gene_data["n"] < 1).any():
        raise ValueError("All dose groups must have n >= 1")

    dataset = pybmds.ContinuousDataset(
        doses=gene_data["dose"].tolist(),
        means=gene_data["mean_log2fc"].tolist(),
        stdevs=gene_data["sd_log2fc"].tolist(),
        ns=gene_data["n"].astype(int).tolist(),
        name=gene_name,
    )

    return dataset
# =============================================================================
# Suppress C++ stdout/stderr noise from bmdscore
# =============================================================================

@contextmanager
def _suppress_output():
    """Redirect stdout and stderr to /dev/null temporarily."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


# =============================================================================
# Single gene fitting (all models)
# =============================================================================

def get_pvalue(model):
    """Extract Test 4 (model fit) p-value."""
    try:
        return model.results.tests.p_values[3]
    except (AttributeError, IndexError):
        return np.nan


def pybmd_fit_gene(gene_data, gene_name, bmr, bmr_type, alpha):
    """
    Fit all default continuous models for one gene.
    Returns a list of dicts, one per model.
    """
    gd = gene_data.sort_values("dose").reset_index(drop=True)

    dataset = pybmds.ContinuousDataset(
        doses=gd["dose"].tolist(),
        means=gd["mean_log2fc"].tolist(),
        stdevs=gd["sd_log2fc"].tolist(),
        ns=gd["n"].astype(int).tolist(),
        name=gene_name,
    )

    # run session
    sys.stdout.flush()
    sys.stderr.flush()
    with _suppress_output(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            session = pybmds.Session(dataset=dataset)
            session.add_default_models(settings={
                "bmr": bmr,
                "bmr_type": bmr_type,
                "alpha": alpha,
            })
            session.execute_and_recommend()

    recommended = session.recommended_model
    recommended_name = recommended.name() if recommended else None

    results = []
    for model in session.models:
        row = {
            "gene": gene_name,
            "model": model.name(),
            "bmd": np.nan,
            "bmdl": np.nan,
            "bmdu": np.nan,
            "aic": np.nan,
            "pvalue": np.nan,
            "converged": False,
            "is_recommended": False,
        }

        if model.results is not None:
            row["bmd"] = model.results.bmd
            row["bmdl"] = model.results.bmdl
            row["bmdu"] = model.results.bmdu
            row["aic"] = model.results.fit.aic if hasattr(model.results, "fit") else np.nan
            row["pvalue"] = get_pvalue(model)
            row["converged"] = True
            row["is_recommended"] = (model.name() == recommended_name)

        results.append(row)

    return results


def pybmd_fit_gene_wrapper(args):
    """Unpacks args for ProcessPoolExecutor."""
    gene_data, gene_name, bmr, bmr_type, alpha = args
    try:
        return pybmd_fit_gene(gene_data, gene_name, bmr, bmr_type, alpha)
    except Exception as e:
        return [{
            "gene": gene_name,
            "model": None,
            "bmd": np.nan,
            "bmdl": np.nan,
            "bmdu": np.nan,
            "aic": np.nan,
            "pvalue": np.nan,
            "converged": False,
            "is_recommended": False,
            "error": str(e),
        }]


# =============================================================================
# Batch fitting
# =============================================================================

def fit_all_genes_all_models(
    bmd_input,
    bmr=1.0,
    bmr_type=ContinuousRiskType.StandardDeviation,
    alpha=0.05,
    n_jobs=1,
    verbose=True,
    progress_interval=100,
):
    """
    Fit all BMD models for every gene.

    Returns a DataFrame with one row per model per gene.
    The best model for each gene has is_recommended=True.

    Args:
        bmd_input: DataFrame with columns [gene, dose, mean_log2fc, sd_log2fc, n]
        bmr: benchmark response value (default 1.0 SD)
        bmr_type: ContinuousRiskType (default StandardDeviation)
        alpha: confidence level (default 0.05)
        n_jobs: parallel workers (1=sequential, -1=all CPUs)
        verbose: print progress
        progress_interval: print every N genes
    """
    genes = bmd_input["gene"].unique()
    n_genes = len(genes)

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    if verbose:
        print(f"Fitting BMD models for {n_genes} genes (n_jobs={n_jobs})...")

    # build per-gene data
    gene_args = [
        (bmd_input[bmd_input["gene"] == gene].copy(), gene, bmr, bmr_type, alpha)
        for gene in genes
    ]

    all_results = []

    if n_jobs == 1:
        # sequential
        for i, args in enumerate(gene_args):
            results = pybmd_fit_gene_wrapper(args)
            all_results.extend(results)
            if verbose and (i + 1) % progress_interval == 0:
                print(f"  {i + 1}/{n_genes} genes done")
    else:
        # parallel
        n_done = 0
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(pybmd_fit_gene_wrapper, args): args[1]
                for args in gene_args
            }
            for future in as_completed(futures):
                all_results.extend(future.result())
                n_done += 1
                if verbose and n_done % progress_interval == 0:
                    print(f"  {n_done}/{n_genes} genes done")

    if verbose:
        print(f"  Complete: {n_genes} genes")

    results_df = pd.DataFrame(all_results)

    # restore original gene order
    gene_order = {g: i for i, g in enumerate(genes)}
    results_df["_order"] = results_df["gene"].map(gene_order)
    results_df = results_df.sort_values(["_order", "model"]).drop(columns=["_order"]).reset_index(drop=True)

    return results_df


# =============================================================================
# Summary
# =============================================================================

def get_best_models(all_models_df):
    """Extract only the recommended model per gene from fit_all_genes_all_models output."""
    best = all_models_df[all_models_df["is_recommended"] == True].copy()

    # genes where no model was recommended
    missing = set(all_models_df["gene"].unique()) - set(best["gene"].unique())
    if missing:
        missing_rows = pd.DataFrame([{
            "gene": g, "model": None, "bmd": np.nan, "bmdl": np.nan,
            "bmdu": np.nan, "aic": np.nan, "pvalue": np.nan,
            "converged": False, "is_recommended": False,
        } for g in missing])
        best = pd.concat([best, missing_rows], ignore_index=True)

    return best.reset_index(drop=True)


def summarize_results(bmd_results):
    """Quick summary stats from BMD results."""
    converged = bmd_results[bmd_results["converged"] == True]
    n_total = len(bmd_results)
    n_conv = len(converged)

    summary = {
        "total_genes": n_total,
        "converged": n_conv,
        "failed": n_total - n_conv,
        "convergence_rate": n_conv / n_total if n_total else 0,
    }

    if n_conv > 0 and "bmd" in converged.columns:
        bmd_vals = converged["bmd"].dropna()
        summary["bmd_median"] = bmd_vals.median()
        summary["bmd_mean"] = bmd_vals.mean()
        summary["bmd_min"] = bmd_vals.min()
        summary["bmd_max"] = bmd_vals.max()

    if n_conv > 0 and "model" in converged.columns:
        summary["model_counts"] = converged["model"].value_counts().to_dict()

    return summary