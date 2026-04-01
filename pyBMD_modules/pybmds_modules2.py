"""
pyBMDS Model Fitting Modules.

Functions for:
    - Creating pyBMDS datasets from summary data
    - Fitting BMD models for individual genes
    - Batch fitting across all genes
    - Extracting and formatting results (full session.to_df() format)

Designed to work with output from pre_pybmds_modules.py
"""

import os
import sys
import warnings
from contextlib import contextmanager
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import pybmds

# Import the enum for BMR type
try:
    from pybmds import ContinuousRiskType
except ImportError:
    try:
        ContinuousRiskType = pybmds.ContinuousRiskType
    except AttributeError:
        raise ImportError(
            "Cannot find ContinuousRiskType in pybmds. "
            "Please check your pybmds version."
        )


# =============================================================================
# Output Suppression (for C++ library warnings)
# =============================================================================

@contextmanager
def suppress_stdout_stderr():
    """
    Suppress both stdout and stderr output (catches C++ library warnings).

    Some C++ libraries print warnings to stdout instead of stderr,
    so we suppress both to be safe.
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    original_stdout_fd = None
    original_stderr_fd = None

    try:
        original_stdout_fd = os.dup(stdout_fd)
        original_stderr_fd = os.dup(stderr_fd)

        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)

        yield
    finally:
        if original_stdout_fd is not None:
            os.dup2(original_stdout_fd, stdout_fd)
            os.close(original_stdout_fd)
        if original_stderr_fd is not None:
            os.dup2(original_stderr_fd, stderr_fd)
            os.close(original_stderr_fd)


# Backwards compatibility alias
suppress_stderr = suppress_stdout_stderr


# =============================================================================
# Internal Helpers
# =============================================================================

def _run_session(
    dataset: pybmds.ContinuousDataset,
    bmr: float,
    bmr_type: ContinuousRiskType,
    alpha: float,
    suppress_warnings: bool = True,
) -> pybmds.Session:
    """
    Create, configure, execute, and recommend a pyBMDS session.

    Centralises the session setup logic so that every public function
    uses exactly the same execution path.

    Args:
        dataset: A ready-to-fit ContinuousDataset.
        bmr: Benchmark response value.
        bmr_type: Type of BMR.
        alpha: Significance level for confidence bounds.
        suppress_warnings: Silence C++ / numpy warnings.

    Returns:
        The fully-executed pybmds.Session with recommendations applied.
    """
    def _execute():
        session = pybmds.Session(dataset=dataset)
        session.add_default_models(
            settings={
                "bmr": bmr,
                "bmr_type": bmr_type,
                "alpha": alpha,
            }
        )
        session.execute_and_recommend()
        return session

    if suppress_warnings:
        sys.stdout.flush()
        sys.stderr.flush()
        with suppress_stdout_stderr():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    return _execute()
    else:
        return _execute()


def _extract_pvalue(model) -> float:
    """Extract Test 4 (model fit) p-value from a fitted model."""
    if (
        hasattr(model, "results")
        and model.results is not None
        and hasattr(model.results, "tests")
        and model.results.tests is not None
    ):
        p_values = model.results.tests.p_values
        if len(p_values) > 3:
            return p_values[3]
    return np.nan


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
# Single Gene Fitting
# =============================================================================

def fit_single_gene(
    gene_data: pd.DataFrame,
    gene_name: str,
    bmr: float = 1.0,
    bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,
    alpha: float = 0.05,
    suppress_warnings: bool = True,
) -> Dict[str, Any]:
    """
    Fit BMD models for a single gene and return best model results.

    Fits all default continuous models (Hill, Exponential, Power, Polynomial)
    and selects the best by AIC among models that pass goodness-of-fit.

    Args:
        gene_data: DataFrame with columns [gene, dose, mean_log2fc, sd_log2fc, n]
        gene_name: Gene identifier
        bmr: Benchmark response (default: 1.0 SD change from control)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level for BMDL/BMDU confidence bounds (default: 0.05)
        suppress_warnings: Suppress C++ library warnings (default: True)

    Returns:
        Dictionary with keys: gene, bmd, bmdl, bmdu, model, aic, pvalue,
        converged, error
    """
    if bmr <= 0:
        raise ValueError(f"bmr must be positive, got {bmr}")

    result = {
        "gene": gene_name,
        "bmd": np.nan,
        "bmdl": np.nan,
        "bmdu": np.nan,
        "model": None,
        "aic": np.nan,
        "pvalue": np.nan,
        "converged": False,
        "error": None,
    }

    try:
        dataset = create_continuous_dataset(gene_data, gene_name)
        session = _run_session(dataset, bmr, bmr_type, alpha, suppress_warnings)
        best = session.recommended_model

        if best is not None and hasattr(best, "results") and best.results is not None:
            result.update({
                "bmd": best.results.bmd,
                "bmdl": best.results.bmdl,
                "bmdu": best.results.bmdu,
                "model": best.name(),
                "aic": (
                    best.results.fit.aic
                    if hasattr(best.results, "fit")
                    else np.nan
                ),
                "pvalue": _extract_pvalue(best),
                "converged": True,
            })
        else:
            result["error"] = "No valid model fit"

    except Exception as e:
        result["error"] = str(e)

    return result


def fit_single_gene_all_models(
    gene_data: pd.DataFrame,
    gene_name: str,
    bmr: float = 1.0,
    bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,
    alpha: float = 0.05,
    suppress_warnings: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fit all BMD models for a single gene and return results for each.

    Unlike fit_single_gene which returns only the best model,
    this returns results for all fitted models.

    Args:
        gene_data: DataFrame with columns [gene, dose, mean_log2fc, sd_log2fc, n]
        gene_name: Gene identifier
        bmr: Benchmark response (default: 1.0 SD)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level (default: 0.05)
        suppress_warnings: Suppress C++ library warnings (default: True)

    Returns:
        List of dictionaries, one per model attempted.
    """
    if bmr <= 0:
        raise ValueError(f"bmr must be positive, got {bmr}")

    results = []

    try:
        dataset = create_continuous_dataset(gene_data, gene_name)
        session = _run_session(dataset, bmr, bmr_type, alpha, suppress_warnings)
        recommended = session.recommended_model
        recommended_name = recommended.name() if recommended is not None else None

        for model in session.models:
            res = {
                "gene": gene_name,
                "model": model.name(),
                "bmd": np.nan,
                "bmdl": np.nan,
                "bmdu": np.nan,
                "aic": np.nan,
                "pvalue": np.nan,
                "converged": False,
                "is_recommended": False,
                "error": None,
            }

            if hasattr(model, "results") and model.results is not None:
                res.update({
                    "bmd": model.results.bmd,
                    "bmdl": model.results.bmdl,
                    "bmdu": model.results.bmdu,
                    "aic": (
                        model.results.fit.aic
                        if hasattr(model.results, "fit")
                        else np.nan
                    ),
                    "pvalue": _extract_pvalue(model),
                    "converged": True,
                    "is_recommended": (model.name() == recommended_name),
                })

            results.append(res)

    except Exception as e:
        results.append({
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
        })

    return results


# =============================================================================
# Full Output Functions  (session.to_df() format — one row per gene)
# =============================================================================

def fit_single_gene_full_output(
    gene_data: pd.DataFrame,
    gene_name: str,
    bmr: float = 1.0,
    bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,
    alpha: float = 0.05,
    suppress_warnings: bool = True,
) -> pd.DataFrame:
    """
    Fit BMD models for a single gene and return the full session.to_df()
    output for the recommended (best) model only.

    This produces the same rich output that you would get from:

        session = pybmds.Session(dataset=dataset)
        session.add_default_models()
        session.execute_and_recommend()
        df = session.to_df()          # <-- same format

    but filtered to the single best model and tagged with gene name.

    Args:
        gene_data: DataFrame with columns [dose, mean_log2fc, sd_log2fc, n]
        gene_name: Gene identifier
        bmr: Benchmark response (default: 1.0 SD)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level (default: 0.05)
        suppress_warnings: Suppress C++ library warnings (default: True)

    Returns:
        DataFrame with full model output (single row for best model).
        Returns a minimal error DataFrame if fitting fails.
    """
    if bmr <= 0:
        raise ValueError(f"bmr must be positive, got {bmr}")

    try:
        dataset = create_continuous_dataset(gene_data, gene_name)
        session = _run_session(dataset, bmr, bmr_type, alpha, suppress_warnings)

        # Full DataFrame from pyBMDS (one row per model)
        df = session.to_df()

        # Keep recommended model only
        if "recommended" in df.columns:
            best_df = df[df["recommended"] == True].copy()
        else:
            best_df = df.head(1).copy()

        # Prepend gene name
        best_df.insert(0, "gene", gene_name)

        return best_df

    except Exception as e:
        return pd.DataFrame([{
            "gene": gene_name,
            "error": str(e),
            "converged": False,
        }])


# =============================================================================
# Batch Fitting — Full Output (recommended for most users)
# =============================================================================

def _fit_single_gene_full_output_wrapper(args):
    """Wrapper for parallel processing (must be module-level for pickling)."""
    gene_data, gene_name, bmr, bmr_type, alpha = args
    return fit_single_gene_full_output(
        gene_data, gene_name, bmr, bmr_type, alpha, suppress_warnings=True
    )


def fit_all_genes_full_output(
    bmd_input: pd.DataFrame,
    bmr: float = 1.0,
    bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,
    alpha: float = 0.05,
    verbose: bool = True,
    progress_interval: int = 100,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Fit BMD models for every gene and return **full session.to_df() output**
    for the best (recommended) model per gene.

    This is the primary batch function.  Each gene gets one row whose
    columns match the native ``session.to_df()`` output, plus a leading
    ``gene`` column.

    Equivalent to looping over genes and running:

        session = pybmds.Session(dataset=dataset)
        session.add_default_models()
        session.execute_and_recommend()
        df = session.to_df()   # <-- you get these columns
        df.to_excel("report.xlsx")

    Args:
        bmd_input: Long-format DataFrame [gene, dose, mean_log2fc, sd_log2fc, n]
        bmr: Benchmark response (default: 1.0 SD)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level (default: 0.05)
        verbose: Print progress (default: True)
        progress_interval: Print every N genes (default: 100)
        n_jobs: Parallel workers (1 = sequential, -1 = all CPUs)

    Returns:
        DataFrame — one row per gene, full pyBMDS output for the best model.
    """
    genes = bmd_input["gene"].unique()
    n_genes = len(genes)

    if verbose:
        print(f"Fitting BMD models for {n_genes} genes (n_jobs={n_jobs})...")

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    all_results: List[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Sequential
    # ------------------------------------------------------------------
    if n_jobs == 1:
        for i, gene in enumerate(genes):
            gene_data = bmd_input[bmd_input["gene"] == gene]
            result_df = fit_single_gene_full_output(
                gene_data, gene, bmr, bmr_type, alpha, suppress_warnings=True
            )
            all_results.append(result_df)

            if verbose and (i + 1) % progress_interval == 0:
                print(f"  Processed {i + 1}/{n_genes} genes")

        if verbose:
            print(f"  Complete: {n_genes} genes")

        return pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # Parallel
    # ------------------------------------------------------------------
    from concurrent.futures import ProcessPoolExecutor, as_completed

    gene_data_list = [
        (bmd_input[bmd_input["gene"] == gene].copy(), gene, bmr, bmr_type, alpha)
        for gene in genes
    ]

    n_completed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_gene = {
            executor.submit(_fit_single_gene_full_output_wrapper, args): args[1]
            for args in gene_data_list
        }
        for future in as_completed(future_to_gene):
            gene_name = future_to_gene[future]
            try:
                all_results.append(future.result())
            except Exception as e:
                all_results.append(pd.DataFrame([{
                    "gene": gene_name,
                    "error": str(e),
                    "converged": False,
                }]))

            n_completed += 1
            if verbose and n_completed % progress_interval == 0:
                print(f"  Processed {n_completed}/{n_genes} genes")

    if verbose:
        print(f"  Complete: {n_genes} genes")

    results_df = pd.concat(all_results, ignore_index=True)
    gene_order = {g: i for i, g in enumerate(genes)}
    results_df["_order"] = results_df["gene"].map(gene_order)
    results_df = (
        results_df.sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    return results_df


# =============================================================================
# Batch Fitting — Simplified Output
# =============================================================================

def _fit_single_gene_wrapper(args):
    """Wrapper for parallel processing (must be module-level for pickling)."""
    gene_data, gene_name, bmr, bmr_type, alpha = args
    return fit_single_gene(
        gene_data, gene_name, bmr, bmr_type, alpha, suppress_warnings=True
    )


def _fit_single_gene_all_models_wrapper(args):
    """Wrapper for parallel processing of all models."""
    gene_data, gene_name, bmr, bmr_type, alpha = args
    return fit_single_gene_all_models(
        gene_data, gene_name, bmr, bmr_type, alpha, suppress_warnings=True
    )


def fit_all_genes(
    bmd_input: pd.DataFrame,
    bmr: float = 1.0,
    bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,
    alpha: float = 0.05,
    verbose: bool = True,
    progress_interval: int = 100,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Fit BMD models for all genes — simplified output (one row per gene).

    Returns a compact DataFrame with: gene, bmd, bmdl, bmdu, model, aic,
    pvalue, converged, error.

    For the full session.to_df() output use ``fit_all_genes_full_output``.

    Args:
        bmd_input: Long-format DataFrame [gene, dose, mean_log2fc, sd_log2fc, n]
        bmr: Benchmark response (default: 1.0 SD)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level (default: 0.05)
        verbose: Print progress (default: True)
        progress_interval: Print every N genes (default: 100)
        n_jobs: Parallel workers (1 = sequential, -1 = all CPUs)

    Returns:
        DataFrame with one row per gene (best model only).
    """
    genes = bmd_input["gene"].unique()
    n_genes = len(genes)

    if verbose:
        print(f"Fitting BMD models for {n_genes} genes (n_jobs={n_jobs})...")

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # ------------------------------------------------------------------
    # Sequential
    # ------------------------------------------------------------------
    if n_jobs == 1:
        results = []
        n_failed = 0

        for i, gene in enumerate(genes):
            gene_data = bmd_input[bmd_input["gene"] == gene]
            res = fit_single_gene(
                gene_data, gene, bmr, bmr_type, alpha, suppress_warnings=True
            )
            results.append(res)
            if not res["converged"]:
                n_failed += 1

            if verbose and (i + 1) % progress_interval == 0:
                print(
                    f"  Processed {i + 1}/{n_genes} genes "
                    f"({n_failed} failed so far)"
                )

        if verbose:
            n_success = n_genes - n_failed
            pct = 100 * n_success / n_genes if n_genes else 0
            print(f"  Complete: {n_success}/{n_genes} converged ({pct:.1f}%)")

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Parallel
    # ------------------------------------------------------------------
    from concurrent.futures import ProcessPoolExecutor, as_completed

    gene_data_list = [
        (bmd_input[bmd_input["gene"] == gene].copy(), gene, bmr, bmr_type, alpha)
        for gene in genes
    ]

    results = []
    n_completed = 0
    n_failed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_gene = {
            executor.submit(_fit_single_gene_wrapper, args): args[1]
            for args in gene_data_list
        }
        for future in as_completed(future_to_gene):
            gene_name = future_to_gene[future]
            try:
                res = future.result()
                results.append(res)
                if not res["converged"]:
                    n_failed += 1
            except Exception as e:
                results.append({
                    "gene": gene_name,
                    "bmd": np.nan,
                    "bmdl": np.nan,
                    "bmdu": np.nan,
                    "model": None,
                    "aic": np.nan,
                    "pvalue": np.nan,
                    "converged": False,
                    "error": str(e),
                })
                n_failed += 1

            n_completed += 1
            if verbose and n_completed % progress_interval == 0:
                print(
                    f"  Processed {n_completed}/{n_genes} genes "
                    f"({n_failed} failed so far)"
                )

    if verbose:
        n_success = n_genes - n_failed
        pct = 100 * n_success / n_genes if n_genes else 0
        print(f"  Complete: {n_success}/{n_genes} converged ({pct:.1f}%)")

    results_df = pd.DataFrame(results)
    gene_order = {g: i for i, g in enumerate(genes)}
    results_df["_order"] = results_df["gene"].map(gene_order)
    results_df = (
        results_df.sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    return results_df


def fit_all_genes_all_models(
    bmd_input: pd.DataFrame,
    bmr: float = 1.0,
    bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,
    alpha: float = 0.05,
    verbose: bool = True,
    progress_interval: int = 100,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Fit all BMD models for all genes (returns every model, not just best).

    Each gene will have multiple rows (one per model), with
    ``is_recommended=True`` marking the best.

    Args:
        bmd_input: Long-format DataFrame [gene, dose, mean_log2fc, sd_log2fc, n]
        bmr: Benchmark response (default: 1.0 SD)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level (default: 0.05)
        verbose: Print progress (default: True)
        progress_interval: Print every N genes (default: 100)
        n_jobs: Parallel workers (1 = sequential, -1 = all CPUs)

    Returns:
        DataFrame with all model results (multiple rows per gene).
    """
    genes = bmd_input["gene"].unique()
    n_genes = len(genes)

    if verbose:
        print(f"Fitting all BMD models for {n_genes} genes (n_jobs={n_jobs})...")

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # ------------------------------------------------------------------
    # Sequential
    # ------------------------------------------------------------------
    if n_jobs == 1:
        all_results: List[Dict] = []

        for i, gene in enumerate(genes):
            gene_data = bmd_input[bmd_input["gene"] == gene]
            gene_results = fit_single_gene_all_models(
                gene_data, gene, bmr, bmr_type, alpha, suppress_warnings=True
            )
            all_results.extend(gene_results)

            if verbose and (i + 1) % progress_interval == 0:
                print(f"  Processed {i + 1}/{n_genes} genes")

        if verbose:
            print(f"  Complete: {n_genes} genes")

        return pd.DataFrame(all_results)

    # ------------------------------------------------------------------
    # Parallel
    # ------------------------------------------------------------------
    from concurrent.futures import ProcessPoolExecutor, as_completed

    gene_data_list = [
        (bmd_input[bmd_input["gene"] == gene].copy(), gene, bmr, bmr_type, alpha)
        for gene in genes
    ]

    all_results: List[Dict] = []
    n_completed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_gene = {
            executor.submit(_fit_single_gene_all_models_wrapper, args): args[1]
            for args in gene_data_list
        }
        for future in as_completed(future_to_gene):
            gene_name = future_to_gene[future]
            try:
                all_results.extend(future.result())
            except Exception as e:
                all_results.append({
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
                })

            n_completed += 1
            if verbose and n_completed % progress_interval == 0:
                print(f"  Processed {n_completed}/{n_genes} genes")

    if verbose:
        print(f"  Complete: {n_genes} genes")

    results_df = pd.DataFrame(all_results)
    gene_order = {g: i for i, g in enumerate(genes)}
    results_df["_order"] = results_df["gene"].map(gene_order)
    results_df = (
        results_df.sort_values(["_order", "model"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    return results_df


# =============================================================================
# Result Utilities
# =============================================================================

def get_best_models(all_models_results: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the best (recommended) model for each gene.

    Args:
        all_models_results: DataFrame from fit_all_genes_all_models()

    Returns:
        DataFrame with one row per gene (the recommended model).
    """
    best = all_models_results[all_models_results["is_recommended"] == True].copy()

    all_genes = set(all_models_results["gene"].unique())
    best_genes = set(best["gene"].unique())
    missing_genes = all_genes - best_genes

    if missing_genes:
        missing_rows = [
            {
                "gene": gene,
                "model": None,
                "bmd": np.nan,
                "bmdl": np.nan,
                "bmdu": np.nan,
                "aic": np.nan,
                "pvalue": np.nan,
                "converged": False,
                "is_recommended": False,
                "error": "No model recommended",
            }
            for gene in missing_genes
        ]
        best = pd.concat(
            [best, pd.DataFrame(missing_rows)], ignore_index=True
        )

    return best.reset_index(drop=True)


def summarize_results(bmd_results: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics from BMD results.

    Args:
        bmd_results: DataFrame from fit_all_genes() or
                     fit_all_genes_full_output()

    Returns:
        Dictionary with summary statistics.
    """
    converged = bmd_results[bmd_results["converged"] == True]
    n_total = len(bmd_results)
    n_conv = len(converged)

    summary: Dict[str, Any] = {
        "total_genes": n_total,
        "n_converged": n_conv,
        "n_failed": n_total - n_conv,
        "convergence_rate": n_conv / n_total if n_total > 0 else 0,
    }

    # BMD distribution (use 'bmd' column — present in both simplified and
    # full outputs)
    bmd_col = "bmd" if "bmd" in converged.columns else None
    if bmd_col and n_conv > 0:
        bmd_vals = converged[bmd_col].dropna()
        summary.update({
            "bmd_median": bmd_vals.median(),
            "bmd_mean": bmd_vals.mean(),
            "bmd_std": bmd_vals.std(),
            "bmd_min": bmd_vals.min(),
            "bmd_max": bmd_vals.max(),
            "bmd_q25": bmd_vals.quantile(0.25),
            "bmd_q75": bmd_vals.quantile(0.75),
        })

    # Model distribution
    model_col = "model" if "model" in converged.columns else None
    if model_col and n_conv > 0:
        model_counts = converged[model_col].value_counts().to_dict()
        summary["model_distribution"] = model_counts
        summary["most_common_model"] = (
            converged[model_col].mode().iloc[0]
        )

    return summary


def filter_results(
    bmd_results: pd.DataFrame,
    max_bmd: Optional[float] = None,
    min_pvalue: Optional[float] = None,
    models: Optional[List[str]] = None,
    keep_failed: bool = False,
) -> pd.DataFrame:
    """
    Filter BMD results based on criteria.

    Args:
        bmd_results: DataFrame from fit_all_genes()
        max_bmd: Maximum BMD value to include
        min_pvalue: Minimum goodness-of-fit p-value (exclude poor fits)
        models: List of model names to include
        keep_failed: If True, retain non-converged genes in output
                     (default: False)

    Returns:
        Filtered DataFrame.
    """
    if keep_failed:
        filtered = bmd_results.copy()
    else:
        filtered = bmd_results[bmd_results["converged"] == True].copy()

    if max_bmd is not None:
        filtered = filtered[filtered["bmd"] <= max_bmd]

    if min_pvalue is not None:
        filtered = filtered[filtered["pvalue"] >= min_pvalue]

    if models is not None:
        filtered = filtered[filtered["model"].isin(models)]

    return filtered