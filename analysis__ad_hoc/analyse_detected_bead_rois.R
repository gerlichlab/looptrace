library("argparse")
library("data.table")
library("ggplot2")

cli_parser <- ArgumentParser(description="Visualise bead ROI detection results")
cli_parser$add_argument("-i", "--input-file", required=TRUE, help="Path to the fits file created by drift_correct_analysis.py")
cli_parser$add_argument("-o", "--output-file", required=TRUE, help="Path to folder in which to place output files")
cli_parser$add_argument("--do-not-modify-counts", action="store_true", help="Indicate that counts are real counts not line counts (no header, e.g.)")
opts <- cli_parser$parse_args()

message("Reading bead ROI counts: ", opts$input_file)
roi_counts <- fread(opts$infile)
roi_counts$frame <- as.factor(roi_counts$frame)
if (opts$do_not_modify_counts) {
    message("Skipping count modification")
} else {
    message("Deducting one from each count to account for header in line count")
    roi_counts$count <- roi_counts$count - 1
}
if (any(roi_counts$count < 0)) {
    stop("Counts table has negative count values! ", sum(roi_counts$counts < 0))
}

message("Building per-frame boxplot")
roi_counts_boxplot <- ggplot(roi_counts, aes(x=frame, y=count)) + 
    geom_boxplot() + 
    ggtitle("Detected bead ROI count by frame, across FOVs")

message("Saving plot: ", opts$output_file)
ggsave(filename = opts$output_file, plot = roi_counts_boxplot)
