
import subprocess
import pandas as pd
from pathlib import Path
 
# ---- Config ----
COUNT_FILE = "/home/hanna/Desktop/RnD/toxicology/toxicology_modules/QC_modules/BIOS2931_qc_filtered/counts_qc_passed.csv"
ANNOTATION_FILE = "/home/hanna/Desktop/RnD/toxicology/toxicology_modules/DEG_modules/BIOS2931_deg_annot.csv"
RSCRIPT_PATH = "DEG_to_control_dose.r"
OUTPUT_DIR = "deseq2_results"
CONTROL_NAMES = "DMSO1,DMSO2"
 
# ---- Read annotation ----
annot = pd.read_csv(ANNOTATION_FILE)
 
# ---- Get unique cell_line + compound pairs, excluding controls ----
control_list = [c.strip() for c in CONTROL_NAMES.split(",")]
treatments = annot[~annot["compound"].isin(control_list)]
pairs = treatments[["sample", "compound"]].drop_duplicates()
 
# ---- Run DESeq2 for each pair ----
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
 
for _, row in pairs.iterrows():
    cell_line = row["sample"]
    compound = row["compound"]
    output_path = f"{OUTPUT_DIR}/{cell_line}_{compound}"
    Path(output_path).mkdir(parents=True, exist_ok=True)
 
    print(f"Running: {cell_line} | {compound}")
 
    subprocess.run([
        "Rscript", RSCRIPT_PATH,
        COUNT_FILE,
        ANNOTATION_FILE,
        cell_line,
        compound,
        CONTROL_NAMES,
        output_path,
    ], check=True)
 
print("Done!")