#pybmds_modules.py

# pybmds_modules.py
"""
pyBMDS Model Fitting Modules.

Functions for:
    - Creating pyBMDS datasets from summary data
    - Fitting BMD models for individual genes
    - Batch fitting across all genes
    - Extracting and formatting results

Designed to work with output from pre_pybmds_modules.py
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import pybmds
from pybmds.types.continuous import ContinuousModelSettings, ContinuousRiskType


# =============================================================================
# Dataset Creation
# =============================================================================

def create_continuous_dataset(gene_data, gene_name ) : 
    """
    Create a pyBMDS ContinuousDataset for a single gene.
    
    Args:
        gene_data: DataFrame with columns [dose, mean_log2fc, sd_log2fc, n]
        gene_name: Gene identifier for labeling
    
    Returns:
        pybmds.ContinuousDataset ready for model fitting
    
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    required_cols = ["dose", "mean_log2fc", "sd_log2fc", "n"]
    missing = set(required_cols) - set(gene_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Sort by dose to ensure correct order
    gene_data = gene_data.sort_values("dose").reset_index(drop=True)
    
    # Validate data
    if len(gene_data) < 3:
        raise ValueError(f"Need at least 3 dose groups, got {len(gene_data)}")
    
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

def fit_single_gene(gene_data , gene_name, bmr =1.0 , bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,  alpha: float = 0.05 ) :

    """
    Fit BMD models for a single gene and return best model results.
    
    Fits all default continuous models (Hill, Exponential, Power, Polynomial)
    and selects the best by AIC among models that pass goodness-of-fit.
    
    Args:
        gene_data: DataFrame with columns [dose, mean_log2fc, sd_log2fc, n]
        gene_name: Gene identifier
        bmr: Benchmark response (default: 1.0 SD change from control)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level for BMDL/BMDU confidence bounds (default: 0.05)
    
    Returns:
        Dictionary with keys:
            - gene: Gene name
            - bmd: Benchmark dose estimate
            - bmdl: Lower confidence bound
            - bmdu: Upper confidence bound
            - model: Best model name
            - aic: AIC of best model
            - pvalue: Goodness-of-fit p-value
            - converged: Whether fitting succeeded
            - error: Error message if failed
    """
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
        # Create dataset
        dataset = create_continuous_dataset(gene_data, gene_name)
        
        # Create session (no settings argument - settings go on add_default_models)
        session = pybmds.Session(dataset=dataset)
        
        # Add models with settings
        session.add_default_models(
            settings={
                "bmr": bmr,
                "bmr_type": bmr_type,
                "alpha": alpha,
            }
        )
        
        # Execute and recommend best model
        session.execute()
        session.recommend()
        
        # Get recommended model
        best = session.recommended_model
        
        if best is not None and hasattr(best, "results") and best.results is not None:
            # Extract p-value from Test 4 (model fit test)
            pvalue = np.nan
            if hasattr(best.results, "tests") and best.results.tests is not None:
                p_values = best.results.tests.p_values
                if len(p_values) > 3:
                    pvalue = p_values[3]  # Test 4 p-value
            
            result.update({
                "bmd": best.results.bmd,
                "bmdl": best.results.bmdl,
                "bmdu": best.results.bmdu,
                "model": best.name(),
                "aic": best.results.fit.aic if hasattr(best.results, "fit") else np.nan,
                "pvalue": pvalue,
                "converged": True,
            })
        else:
            result["error"] = "No valid model fit"
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def fit_single_gene_all_models(gene_data, gene_name , bmr = 1.0 , bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation, alpha: float = 0.05) : 

    """
    Fit all BMD models for a single gene and return results for each.
    
    Unlike fit_single_gene which returns only the best model,
    this returns results for all fitted models.
    
    Args:
        gene_data: DataFrame with columns [dose, mean_log2fc, sd_log2fc, n]
        gene_name: Gene identifier
        bmr: Benchmark response (default: 1.0 SD)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level (default: 0.05)
    
    Returns:
        List of dictionaries, one per model attempted
    """
    results = []
    
    try:
        dataset = create_continuous_dataset(gene_data, gene_name)
        
        settings = ContinuousModelSettings(
            bmr=bmr,
            bmr_type=bmr_type,
            alpha=alpha,
        )
        
        session = pybmds.Session(dataset=dataset, settings=settings)
        session.add_default_models()
        session.execute()
        
        for model in session.models:
            res = {
                "gene": gene_name,
                "model": model.name,
                "bmd": np.nan,
                "bmdl": np.nan,
                "bmdu": np.nan,
                "aic": np.nan,
                "pvalue": np.nan,
                "converged": False,
                "is_recommended": False,
                "error": None,
            }
            
            if model.has_results:
                res.update({
                    "bmd": model.results.bmd,
                    "bmdl": model.results.bmdl,
                    "bmdu": model.results.bmdu,
                    "aic": model.results.aic,
                    "pvalue": getattr(model.results.gof, "p_value", np.nan) if hasattr(model.results, "gof") else np.nan,
                    "converged": True,
                })
            
            results.append(res)
        
        # Mark recommended model
        recommended = session.recommend()
        if recommended is not None:
            for res in results:
                if res["model"] == recommended.name:
                    res["is_recommended"] = True
                    break
                    
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
# Batch Fitting
# =============================================================================

def fit_all_genes(bmd_input, bmr: float = 1.0,bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,alpha: float = 0.05, verbose = True, progress_interval: int = 100) : 

    """
    Fit BMD models for all genes in the input data.
    
    Args:
        bmd_input: Long-format DataFrame with columns [gene, dose, mean_log2fc, sd_log2fc, n]
        bmr: Benchmark response (default: 1.0 SD)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level (default: 0.05)
        verbose: Print progress (default: True)
        progress_interval: Print progress every N genes (default: 100)
    
    Returns:
        DataFrame with BMD results for all genes (one row per gene, best model only)
    """
    genes = bmd_input["gene"].unique()
    n_genes = len(genes)
    
    if verbose:
        print(f"Fitting BMD models for {n_genes} genes...")
    
    results = []
    n_failed = 0
    
    for i, gene in enumerate(genes):
        gene_data = bmd_input[bmd_input["gene"] == gene].copy()
        
        # Suppress warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_single_gene(gene_data, gene, bmr, bmr_type, alpha)
        
        results.append(res)
        
        if not res["converged"]:
            n_failed += 1
        
        # Progress update
        if verbose and (i + 1) % progress_interval == 0:
            print(f"  Processed {i + 1}/{n_genes} genes ({n_failed} failed so far)")
    
    if verbose:
        n_success = n_genes - n_failed
        print(f"  Complete: {n_success}/{n_genes} converged ({100 * n_success / n_genes:.1f}%)")
    
    return pd.DataFrame(results)


def fit_all_genes_all_models(bmd_input, bmr = 1.0,bmr_type: ContinuousRiskType = ContinuousRiskType.StandardDeviation,alpha: float = 0.05, verbose = True) : 

    """
    Fit all BMD models for all genes (returns all models, not just best).
    
    Use this when you need to compare model fits or want full diagnostics.
    
    Args:
        bmd_input: Long-format DataFrame with columns [gene, dose, mean_log2fc, sd_log2fc, n]
        bmr: Benchmark response (default: 1.0 SD)
        bmr_type: Type of BMR (default: StandardDeviation)
        alpha: Significance level (default: 0.05)
        verbose: Print progress (default: True)
    
    Returns:
        DataFrame with all model results (multiple rows per gene)
    """
    genes = bmd_input["gene"].unique()
    n_genes = len(genes)
    
    if verbose:
        print(f"Fitting all BMD models for {n_genes} genes...")
    
    all_results = []
    
    for i, gene in enumerate(genes):
        gene_data = bmd_input[bmd_input["gene"] == gene].copy()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gene_results = fit_single_gene_all_models(gene_data, gene, bmr, bmr_type, alpha)
        
        all_results.extend(gene_results)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_genes} genes")
    
    if verbose:
        print(f"  Complete: {n_genes} genes")
    
    return pd.DataFrame(all_results)


# =============================================================================
# Result Utilities
# =============================================================================

def summarize_results(bmd_results: pd.DataFrame) :
    """
    Generate summary statistics from BMD results.
    
    Args:
        bmd_results: DataFrame from fit_all_genes()
    
    Returns:
        Dictionary with summary statistics
    """
    converged = bmd_results[bmd_results["converged"]]
    failed = bmd_results[~bmd_results["converged"]]
    
    summary = {
        "total_genes": len(bmd_results),
        "n_converged": len(converged),
        "n_failed": len(failed),
        "convergence_rate": len(converged) / len(bmd_results) if len(bmd_results) > 0 else 0,
        "bmd_median": converged["bmd"].median() if len(converged) > 0 else np.nan,
        "bmd_mean": converged["bmd"].mean() if len(converged) > 0 else np.nan,
        "bmd_std": converged["bmd"].std() if len(converged) > 0 else np.nan,
        "bmd_min": converged["bmd"].min() if len(converged) > 0 else np.nan,
        "bmd_max": converged["bmd"].max() if len(converged) > 0 else np.nan,
        "bmd_q25": converged["bmd"].quantile(0.25) if len(converged) > 0 else np.nan,
        "bmd_q75": converged["bmd"].quantile(0.75) if len(converged) > 0 else np.nan,
    }
    
    # Model distribution
    if len(converged) > 0:
        model_counts = converged["model"].value_counts().to_dict()
        summary["model_distribution"] = model_counts
        summary["most_common_model"] = converged["model"].mode().iloc[0] if len(converged) > 0 else None
    
    return summary


def filter_results(
    bmd_results: pd.DataFrame,
    max_bmd: Optional[float] = None,
    min_pvalue: Optional[float] = None,
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter BMD results based on criteria.
    
    Args:
        bmd_results: DataFrame from fit_all_genes()
        max_bmd: Maximum BMD value to include
        min_pvalue: Minimum goodness-of-fit p-value (exclude poor fits)
        models: List of model names to include
    
    Returns:
        Filtered DataFrame
    """
    filtered = bmd_results[bmd_results["converged"]].copy()
    
    if max_bmd is not None:
        filtered = filtered[filtered["bmd"] <= max_bmd]
    
    if min_pvalue is not None:
        filtered = filtered[filtered["pvalue"] >= min_pvalue]
    
    if models is not None:
        filtered = filtered[filtered["model"].isin(models)]
    
    return filtered