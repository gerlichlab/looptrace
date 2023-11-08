# Simple program to do quality control analysis of detected bead ROIs

library("argparse")
library("data.table")
library("ggplot2")
library("stringi")

# CLI definition and parsing
## Parser
cli_parser <- ArgumentParser(description="Visualise bead ROI detection results")
## Required
cli_parser$add_argument("-i", "--input-folder", required = TRUE, help = "Path to folder with detected bead ROIs, 1 file per combination of FOV and hybridistaion round / timepoint")
cli_parser$add_argument("-o", "--output-folder", required = TRUE, help = "Path to folder in which to place output files")
cli_parser$add_argument("--num-positions", type = "integer", required = TRUE, help="Number of positions (fields of view) for the experiment, used for validation of found files collection")
cli_parser$add_argument("--num-frames", type = "integer", required = TRUE, help="Number of frames (hybridisation rounds / timepoints) for the experiment, used for validation of found files collection")
## Optional
cli_parser$add_argument("--counts-files-prefix", default = "bead_rois__", help = "Prefix for files to find to count ROIs")
cli_parser$add_argument("--counts-files-extension", default = "csv", help = "Extension for files to fine to count ROIs")
cli_parser$add_argument("--do-not-modify-counts", action = "store_true", help = "Indicate that counts are real counts not line counts (no header, e.g.)")
cli_parser$add_argument("--position-frame-delimiter", default = "_", help = "Delimiter between position and frame in filename")
opts <- cli_parser$parse_args()

# String maniplulation helper functions
## Remove the given prefix from the given target.
stripPrefix <- function(prefix, target) {
    stringi::stri_replace_first_fixed(str = target, pattern = prefix, replacement = "")
}
## Remove the given suffix from the given target.
stripSuffix <- function(suffix, target) {
    stringi::stri_replace_last_fixed(str = target, pattern = suffix, replacement = "")
}
## Parse field of view (FOV / position) and frame / hybridisation timepoint/round from given filename, based 
## on given prefix and extension at command-line, and delimiter assumed or given between position and frame in filename.
parsePositionAndFrame <- function(fn, file_prefix, file_ext, pos_frame_sep) {
    encoded <- stripPrefix(prefix = file_prefix, target = fn)
    encoded <- stripSuffix(suffix = paste0(".", file_ext), target = encoded)
    fields <- unlist(strsplit(encoded, pos_frame_sep))
    if (length(fields) != 2) {
        stop(sprintf("Failed to parse position and frame from filename: %s. %s field(s): %s", fn, length(fields), paste0(fields, collapse = ",")))
    }
    list(position = as.integer(fields[1]), frame = as.integer(fields[2]))
}

# Build the table.
pattern <- sprintf("%s*.%s", opts$counts_files_prefix, opts$counts_files_extension)
count_rois_cmd <- sprintf("wc -l %s", file.path(opts$input_folder, pattern))
message("Building table from command: ", count_rois_cmd)
roi_counts <- data.table(read.table(text = system(count_rois_cmd, intern = TRUE), stringsAsFactors = FALSE), stringsAsFactors = FALSE)
colnames(roi_counts) <- c("count", "filename")
message("Printing ROI counts table")
roi_counts
roi_counts$filename <- sapply(roi_counts$filename, basename)

# Validate the table.
if (1 != nrow(roi_counts[filename == "total", ])) {
    stop("Expected exactly one row with filename of 'total', but got ", nrow(roi_counts[filename == "total", ]))
}
roi_counts <- roi_counts[filename != "total", ]
## TODO: could validate here not just on total row count as proudct of count of field of view and number of timepoints, 
##       but also validate that hierarchy / particular counts, too.
exp_rows_counts <- opts$num_positions * opts$num_frames
if (nrow(roi_counts) != exp_rows_counts) {
    stop(sprintf("For %s positions and %s frames, %s counts are expected, but got %s", opts$num_positions, opts$num_frames, exp_rows_counts, nrow(roi_counts)))
}
if (any(!startsWith(roi_counts$filename, opts$counts_files_prefix))) {
    stop("Number of rows with invalid prefix: ", sum(!startsWith(roi_counts$filename, opts$counts_files_prefix)))
}
if (any(!endsWith(roi_counts$filename, opts$counts_files_extension))) {
    stop("Number of rows with invalid extension: ", sum(!endsWith(roi_counts$filename, opts$counts_files_extension)))
}

# Update the table.
## Decrement counts to account for presence of header in each line-counted file.
if (opts$do_not_modify_counts) {
    message("Skipping count modification")
} else {
    message("Deducting one from each count to account for header in line count")
    roi_counts$count <- roi_counts$count - 1
}
if (any(roi_counts$count < 0)) {
    stop("Counts table has negative count values! ", sum(roi_counts$count < 0))
}
## Print the validated table.
message("Printing validated table")
roi_counts
## Parse position and frame from each filename.
pos_and_frame <- lapply(roi_counts$filename, function(fn) {
    parsePositionAndFrame(
        fn = fn, 
        file_prefix = opts$counts_files_prefix, 
        file_ext = opts$counts_files_extension, 
        pos_frame_sep = opts$position_frame_delimiter
        )
    })
roi_counts$position <- sapply(pos_and_frame, function(e) e$position)
roi_counts$frame <- sapply(pos_and_frame, function(e) e$frame)
## Make hybridisation timepoint / frame a factor variable rather than integer.

data_output_file <- file.path(opts$output_folder, "bead_roi_counts.csv")
message("Writing bead ROI counts data: ", data_output_file)
write.table(roi_counts, file = data_output_file, quote = FALSE, sep = ",", row.names = FALSE, col.names = TRUE)

saveCountsPlot <- function(fig, plotTypeName) {
    plotfile <- file.path(opts$output_folder, sprintf("bead_roi_counts.%s.png", plotTypeName))
    message("Saving plot: ", plotfile)
    ggsave(filename = plotfile, plot = fig)
    plotfile
}

message("Building per-frame bead ROI count boxplot, all detected")
roi_counts_boxplot <- ggplot(roi_counts, aes(x = as.factor(frame), y = count)) + 
    geom_boxplot() + 
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    ggtitle("Detected bead ROI count by frame, across FOVs")
saveCountsPlot(fig = roi_counts_boxplot, plotTypeName = "detected.boxplot")

message("Building (frame, FOV) bead ROI count heatmap")
roi_counts_heatmap <- ggplot(roi_counts, aes(x = frame, y = position, fill = count)) + 
    geom_tile() + 
    ggtitle("Counts, all detected bead ROIs") + 
    theme_bw()
saveCountsPlot(fig = roi_counts_heatmap, plotTypeName = "detected.heatmap")

message("Done!")
