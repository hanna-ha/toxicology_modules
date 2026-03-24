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
annot$dose[is.na(annot$dose) | annot$dose == "NaN" | annot$dose == ""] <- "0"

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
# Controls get "control" regardless of their original dose
# Treatment samples keep their dose as the label
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

# ---- Extract results: each dose vs control ----
conditions <- levels(dds$condition)
conditions <- conditions[conditions != "control"]

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (cond in conditions) {
  res <- results(dds, contrast = c("condition", cond, "control"))

  res_df <- as.data.frame(res)
  res_df$gene <- rownames(res_df)
  res_df <- res_df[, c("gene", names(res_df)[names(res_df) != "gene"])]

  out_file <- file.path(
    output_dir,
    paste0(cell_line, "_", compound_name, "_", cond, "_vs_control.csv")
  )
  write.csv(res_df, file = out_file, row.names = FALSE)
  cat("Written:", out_file, "\n")
}