library("argparse")
library("data.table")
library("ggplot2")

cli_parser <- ArgumentParser(description="Visualise bead ROI detection results")
cli_parser$add_argument("-i", "--input-file", required=TRUE, help="Path to the fits file created by drift_correct_analysis.py")
cli_parser$add_argument("-o", "--output-file", required=TRUE, help="Path to folder in which to place output files")
opts <- cli_parser$parse_args()

message("Reading bead ROI counts: ", opts$input_file)
roi_counts <- fread(opts$infile)
roi_counts$frame <- as.factor(roi_counts$frame)

message("Building per-frame boxplot")
roi_counts_boxplot <- ggplot(roi_counts, aes(x=frame, y=count)) + 
    geom_boxplot() + 
    ggtitle("Detected bead ROI count by frame, across FOVs")

message("Saving plot: ", opts$output_file)
ggsave(filename = opts$output_file, plot = roi_counts_boxplot)
