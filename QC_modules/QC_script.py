# QC_script
import shutil
from pathlib import Path

import pandas as pd 
import numpy as np 

import qc_config
import qc_modules


# ----------------------------
# Helper
# ----------------------------
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path, index_col=0)

def get_sample_group(sample_name):
    parts = sample_name.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else sample_name



def main() : 

    # ----------------------------
    # Load Dataset
    # ----------------------------
    try:
        count_df = load_csv(qc_config.count_path)
        map_df = load_csv(qc_config.map_path)
    except Exception as e:
        print(f"[ERROR] Failed to load input files: {e}")
        

    all_samples = set(count_df.columns)
    report_rows = []          # collect per-step results for the report
    cumulative_pass = set(count_df.columns)  # intersection across all steps


    print("\n" + "=" * 60)
    print("   QC Start")
    print("=" * 60)
    print(f"Total samples {len(all_samples)}\n")
    # print(cumulative_pass)
    # ----------------------------
    # QC
    # ----------------------------

    # ── 1. Mapped reads (absolute count) ──────────────────────────────────

    qc1_filtered_samples = qc_modules.filter_by_mapped_metrics(df=map_df, row_name=qc_config.mapped_read ,thr=qc_config.mapped_read_thr)

    cumulative_pass &= set(qc1_filtered_samples)

    # ── 2. Mapping rate (%) ───────────────────────────────────────────────
    qc2_filtered_samples = qc_modules.filter_by_mapped_metrics(df=map_df, row_name=qc_config.map_rate , thr=qc_config.map_rate_thr)

    cumulative_pass &= set(qc2_filtered_samples)


    # 3. probes with at least 5 count 
    qc3_filtered_samples = qc_modules.filter_by_probe_count(df=count_df, min_count= qc_config.probe_min_count , min_probes=qc_config.probe_count)
    cumulative_pass &= set(qc3_filtered_samples)


    # 4. p80 
    qc4_filtered_samples = qc_modules.filter_by_p80(df= count_df, thr = qc_config.p80_thr)
    cumulative_pass &= set(qc4_filtered_samples)

    # 5. geni coefficient 
    qc5_filtered_samples = qc_modules.filter_by_gini(df= count_df, thr= qc_config.gini_thr)
    cumulative_pass &= set(qc5_filtered_samples)


    #  6. tukey outer fences
    if qc_config.use_tukey_filter : 
        qc6_filtered_samples = qc_modules.filter_by_tukey(count_df, qc_config.tukey_thr)
        cumulative_pass &= set(qc6_filtered_samples)

    # ----------------------------
    # Individual QC summary
    # ----------------------------
    print("\n" + "=" * 60)
    print("  INDIVIDUAL QC RESULTS")
    print("=" * 60)
    print(f"  Total samples           : {len(all_samples)}")
    print(f"  Passed all individual QC: {len(cumulative_pass)}")
    print(f"  Failed (any QC step)    : {len(all_samples - cumulative_pass)}")
 
    # ----------------------------
    # Replicate-group validation
    # ----------------------------
    # A sample group only passes if ALL replicates passed.
    # Groups are determined by dropping the last _ field from the sample name.
    final_passed = qc_modules.filter_by_replicate_group(
        all_samples=all_samples,
        passed_samples=cumulative_pass, replicate_separator=qc_config.replicate_separator
    )
 
    # ----------------------------
    # Final summary
    # ----------------------------
    print("\n" + "=" * 60)
    print("  FINAL QC SUMMARY")
    print("=" * 60)
    print(f"  Total samples       : {len(all_samples)}")
    print(f"  Final passed        : {len(final_passed)}")

    print("=" * 60)
 
    # ----------------------------
    # Save outputs
    # ----------------------------
    out_dir = Path(qc_config.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # filtered count matrix
    filtered_count_df = count_df[sorted(final_passed)]
    filtered_count_path = out_dir / "counts_qc_passed.csv"
    filtered_count_df.to_csv(filtered_count_path)
    print(f"\nFiltered counts saved -> {filtered_count_path}")
 
    filtered_mapped_df = map_df[sorted(final_passed)]
    filtered_mapped_path = out_dir / "mapped_qc_passed.csv"
    filtered_mapped_df.to_csv(filtered_mapped_path)
    print(f"\nFiltered counts saved -> {filtered_mapped_path}")
 
 

if __name__ == "__main__":
    main()