library("DESeq2")

args <- commandArgs(trailingOnly = TRUE)
count_file      <- args[1]
annotation_file <- args[2]
cell_line       <- args[3]
compound_name   <- args[4]
control_names   <- args[5]  # comma-separated, e.g. "DMSO1,DMSO2"
output_dir      <- args[6]

# ---- Parse control names ----
control_list <- trimws(unlist(strsplit(control_names, ",")))

# ---- Load data ----
cts <- as.matrix(read.csv(count_file, row.names = 1, check.names = FALSE))
annot <- read.csv(annotation_file, check.names = FALSE)

# ---- Clean dose column ----
annot$dose <- as.character(annot$dose)
annot$dose[is.na(annot$dose) | annot$dose == "NaN" | annot$dose == ""] <- "0"  # as some controls doses can be NaN or empty 

# ---- Flag controls by compound name ----
annot$is_control <- annot$compound %in% control_list

# ---- Subset to this cell line: compound + controls ----
annot_sub <- annot[
  annot$sample == cell_line &
  (annot$compound == compound_name | annot$is_control),
]

if (nrow(annot_sub) < 2) {
  stop(paste("Not enough samples for", cell_line, compound_name))
}

# ---- Build a clean condition label ----
annot_sub$condition <- ifelse(annot_sub$is_control, "control", annot_sub$dose)
annot_sub$condition <- factor(annot_sub$condition)

# ---- Build colData ----
colData <- data.frame(
  row.names = annot_sub$sample_name,
  condition = annot_sub$condition
)

# Subset and reorder count matrix
cts_sub <- cts[, rownames(colData)]


# ---- Run DESeq2 ----
dds <- DESeqDataSetFromMatrix(countData = cts_sub, colData = colData, design = ~ condition)
dds$condition <- relevel(dds$condition, ref = "control")
dds <- DESeq(dds)

# ============================================================
# OUTPUT 1: Normalized count matrix (genes x samples)
# ============================================================
norm_counts <- counts(dds, normalized = TRUE)
norm_counts_df <- as.data.frame(norm_counts)
norm_counts_df$gene <- rownames(norm_counts_df)
# Move gene column to front
norm_counts_df <- norm_counts_df[, c("gene", setdiff(names(norm_counts_df), "gene"))]

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

norm_out <- file.path(output_dir, paste0(cell_line, "_", compound_name, "_normalized_counts.csv"))
write.csv(norm_counts_df, file = norm_out, row.names = FALSE)
cat("Written:", norm_out, "\n")

# ============================================================
# OUTPUT 2: DEG summary table (all doses combined)
# Used downstream to filter genes for BMD analysis (padj < 0.05)
# ============================================================
conditions <- levels(dds$condition)
conditions <- conditions[conditions != "control"]

deg_list <- list()

for (cond in conditions) {
  res <- results(dds, contrast = c("condition", cond, "control"))
  res_df <- as.data.frame(res)
  res_df$gene <- rownames(res_df)
  res_df$dose <- cond
  deg_list[[cond]] <- res_df
}

# Combine all dose results into one table
deg_all <- do.call(rbind, deg_list)
rownames(deg_all) <- NULL

# Reorder columns
deg_all <- deg_all[, c("gene", "dose", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")]

deg_out <- file.path(output_dir, paste0(cell_line, "_", compound_name, "_deg_summary.csv"))
write.csv(deg_all, file = deg_out, row.names = FALSE)
cat("Written:", deg_out, "\n")



cat("Done.\n")