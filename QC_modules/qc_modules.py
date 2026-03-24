# qc_modules.py

import pandas as pd
import numpy as np 


# ----------------------------
# filter by mapped table
# ----------------------------

def filter_by_mapped_metrics(df, thr, row_name) : 
 
 
    if row_name not in df.index:
        raise ValueError(
            f"Row '{row_name}' not found. Available row: {list(df.index)}"
        )


 
    # ── Clean & filter ─────────────────────────────────────────────
    df.loc[row_name].astype(float)

    passed = df.columns[df.loc[row_name] >= thr].tolist()
    failed = df.columns[df.loc[row_name] < thr].tolist()
 
    # ── Summary  ──────────────────────────────────────────
    print(f"[QC] {row_name} filter (>= {thr})")
    print(f"     Passed        : {len(passed)}")
    print(f"     Failed        : {len(failed)}\n")
  
    return passed

# ----------------------------
# min probe count
# ----------------------------

def filter_by_probe_count(df, min_count=5, min_probes=None):

    # ── Clean & filter ─────────────────────────────────────────────

    probe_counts = (df >= min_count).sum(axis=0)
 
    passed = probe_counts[probe_counts >= min_probes].index.tolist()
    failed = probe_counts[probe_counts < min_probes].index.tolist()
 
    # ── Summary  ──────────────────────────────────────────
    print(f"[QC] Probe detection (>= {min_count} counts) >= {min_probes} probes")
    print(f"     Passed        : {len(passed)}")
    print(f"     Failed        : {len(failed)}\n")

    return  passed


# ----------------------------
# p80
# ----------------------------

def calculate_p80(values, signal_percent= 80):
    value_list = sorted(values, reverse=True)
    total_mapped = sum(value_list)
    if total_mapped == 0:
        return 0
    threshold = (total_mapped * signal_percent) / 100
    tot_signal = 0
    num_probes = 0
    while tot_signal < threshold:
        tot_signal += value_list[num_probes]
        num_probes += 1
    return num_probes

def filter_by_p80(df, thr, signal_percent=80):

    p80_values = df.apply(
        lambda col: calculate_p80(col.values, signal_percent), axis=0
    )
 
    passed = p80_values[p80_values >= thr].index.tolist()
    failed = p80_values[p80_values < thr].index.tolist()
 
 
    print(f"[QC] P80 filter ")
    print(f"     Passed        : {len(passed)}")
    print(f"     Failed        : {len(failed)}\n")
    return passed


 
 
# ----------------------------
# gini coefficient
# ----------------------------
 
 
def gini_coefficient(x):
    x = np.asarray(x, dtype=np.float64)
 
    if np.all(x == 0):
        return 0.0
 
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
 
    return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n

def filter_by_gini(df, thr):

    gini_values = df.apply(gini_coefficient, axis=0)
 
    passed = gini_values[gini_values <= thr].index.tolist()
    failed = gini_values[gini_values > thr].index.tolist()

    print(f"[QC] Gini coefficient filter  (threshold: <= {thr})")
    print(f"     Passed        : {len(passed)}")
    print(f"     Failed        : {len(failed)}\n")
 
    return passed

# ----------------------------
# Tukey outer fence
# ----------------------------
def filter_by_tukey(count_df, k=1.5):

    lib_sizes = count_df.sum(axis=0)
    q1 = lib_sizes.quantile(0.25)
    q3 = lib_sizes.quantile(0.75)
    iqr = q3 - q1
 
    lower = q1 - k * iqr
    upper = q3 + k * iqr
 
    mask = (lib_sizes >= lower) & (lib_sizes <= upper)
    passed = lib_sizes[mask].index.tolist()
    failed = lib_sizes[~mask].index.tolist()
 
  
    print(f"\n[QC] Tukey's fences  (interquartile )")
    print(f"     Passed        : {len(passed)}")
    print(f"     Failed        : {len(failed)}\n")
 
    return  passed

# ----------------------------
# Filter Ssamples
# ----------------------------

def get_sample_name(sample_name,replicate_separator):

    parts = sample_name.rsplit(replicate_separator, 1)
    return parts[0] if len(parts) > 1 else sample_name
 
 
def filter_by_replicate_group(all_samples, passed_samples , replicate_separator):

    passed_set = set(passed_samples)
 
    # build groups from ALL samples (not just passed)
    groups = {}
    for s in all_samples:
        grp = get_sample_name(s, replicate_separator)
        groups.setdefault(grp, []).append(s)
 
    passed = []
    failed = []
    passed_groups = []
    failed_groups = []
 
    for grp, members in sorted(groups.items()):
        if all(m in passed_set for m in members):
            passed.extend(members)
            passed_groups.append(grp)
        else:
            failed.extend(members)
            failed_groups.append(grp)
    
    print(f"\n[QC] Replicate-group validation")
    print(f"     Total groups          : {len(groups)}")

    print(f"     Total samples passed  : {len(passed)}")
    print(f"     Total samples removed : {len(failed)}")
 
    return passed