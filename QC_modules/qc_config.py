# qc_config.py


# ----------------------------
# Input data
# ----------------------------

# ---------- Paths ----------
count_path = "input/BIOS2931_gene_counts_synthetic_HH.csv"
map_path = "input/BIOS2931_mapped_unmapped_synthetic_HH.csv"

output_path = "test"


replicate_separator = '_'

# ----------------------------
# Qc parameters
# ----------------------------

# mapped reads 
mapped_read = "mapped"
mapped_read_thr =  300000

# mapping rate
map_rate = "PercentMapped"
map_rate_thr = 50

# number of probes with min 5 count
probe_min_count = 5
probe_count = 5000

# p80
p80_thr = 1000

# geni coefficient 
gini_thr = 0.95

# tukeys fences
use_tukey_filter = False
tukey_thr = 3