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
cli_parser$add_argument("--num-rounds", type = "integer", required = TRUE, help="Number of imaging rounds/timepoints for the experiment, used for validation of found files collection")
## Optional
cli_parser$add_argument("--counts-files-prefix", default = "bead_rois__", help = "Prefix for files to find to count ROIs")
cli_parser$add_argument("--counts-files-extension", default = "csv", help = "Extension for files to fine to count ROIs")
cli_parser$add_argument("--do-not-modify-counts", action = "store_true", help = "Indicate that counts are real counts not line counts (no header, e.g.)")
cli_parser$add_argument("--position-frame-delimiter", default = "_", help = "Delimiter between position and frame in filename")
cli_parser$add_argument("--qc-code-column", type = "integer", default = 7, help="Column index (1-based) of the field with the QC value (empty for QC pass) in each bead ROIs file")
## Parse the CLI arguments.
opts <- cli_parser$parse_args()

kBeadRoisPrefix <- "bead_roi_counts"

# Infer delimiter from the extension given for each counts file.
delimiter <- list(csv = ",", tsv = "\t")[[opts$counts_files_extension]]
if (is.null(delimiter)) { stop("Cannot infer delimiter from file extension: ", opts$counts_files_extension) }

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
parsePositionAndFrame <- function(fn) {
    encoded <- stripPrefix(prefix = opts$counts_files_prefix, target = fn)
    encoded <- stripSuffix(suffix = paste0(".", opts$counts_files_extension), target = encoded)
    fields <- unlist(strsplit(encoded, opts$position_frame_delimiter))
    if (length(fields) != 2) {
        stop(sprintf("Failed to parse position and frame from filename: %s. %s field(s): %s", fn, length(fields), paste0(fields, collapse = ",")))
    }
    list(position = as.integer(fields[1]), frame = as.integer(fields[2]))
}

# Count the number of QC-passing (non-failed) ROIs in a particular file (1 file per FOV + timepoint pair).
countPassingQC <- function(f) {
    fn <- basename(f)
    p_and_f <- parsePositionAndFrame(fn)
    codes <- fread(f, sep = delimiter)[[opts$qc_code_column]]
    n_qc_pass <- sum(codes == "") + sum(is.na(codes))
    list(position = p_and_f[["position"]], frame = p_and_f[["frame"]], filename = fn, count = n_qc_pass)
}

# Write the count of ROIs from each (FOV, timepoint) file, either unfiltered or filtered counts.
writeDataFile <- function(counts_table) {
    data_output_file <- file.path(opts$output_folder, sprintf("%s.csv", kBeadRoisPrefix))
    message(sprintf("Writing bead ROI counts data: %s", data_output_file))
    write.table(counts_table, file = data_output_file, quote = FALSE, sep = ",", row.names = FALSE, col.names = TRUE)
    data_output_file
}

buildUnfilteredHeatmap <- function(counts_table) {
    ggplot(counts_table, aes(x = frame, y = position, fill = count_unfiltered))
}

buildFilteredHeatmap <- function(counts_table) {
    ggplot(counts_table, aes(x = frame, y = position, fill = count_filtered))
}

# Create the heatmap for the given counts data (1 count per (FOV, time) pair).
buildCountsHeatmap <- function(counts_table, counts_type_name) {
    if (counts_type_name == "unfiltered") {
        p <- buildUnfilteredHeatmap(counts_table)
    }
    else if (counts_type_name == "filtered") {
        p <- buildFilteredHeatmap(counts_table)
    }
    else { stop("Unrecognised counts filter type: ", counts_type_name) }
    p <- p + 
        geom_tile() + 
        xlab("timepoint") + 
        scale_y_continuous(breaks = round(seq(0, max(counts_table$position), by = 2), 1)) + 
        ggtitle(sprintf("Counts, %s bead ROIs", counts_type_name)) + 
        theme_bw()
    p
}

# Save a bead ROIs counts plot in the current output folder (by CLI), with a particular prefix for the name to indicate content and type.
saveCountsPlot <- function(fig, plot_type_name) {
    plotfile <- file.path(opts$output_folder, sprintf("%s.%s.png", kBeadRoisPrefix, plot_type_name))
    message("Saving plot: ", plotfile)
    ggsave(filename = plotfile, plot = fig)
    plotfile
}

# Sort by FOV and then timepoint.
setKeyPF <- function(unkeyed) { setkey(unkeyed, position, frame) }

# Build the unfiltered table.
pattern <- sprintf("%s*.%s", opts$counts_files_prefix, opts$counts_files_extension)
count_rois_cmd <- sprintf("wc -l %s", file.path(opts$input_folder, pattern))
message("Building table from command: ", count_rois_cmd)
roi_counts <- fread(cmd = count_rois_cmd, stringsAsFactors = FALSE)
colnames(roi_counts) <- c("count", "filename")
message("Printing ROI counts table (before filepath trimming, enrichment, and sorting)")
roi_counts
roi_counts$filename <- sapply(roi_counts$filename, basename)

# Validate the table.
if (1 != nrow(roi_counts[filename == "total", ])) {
    stop("Expected exactly one row with filename of 'total', but got ", nrow(roi_counts[filename == "total", ]))
}
roi_counts <- roi_counts[filename != "total", ]
## TODO: could validate here not just on total row count as proudct of count of field of view and number of timepoints, 
##       but also validate that hierarchy / particular counts, too.
exp_rows_counts <- opts$num_positions * opts$num_rounds
if (nrow(roi_counts) != exp_rows_counts) {
    stop(sprintf("For %s positions and %s frames, %s counts are expected, but got %s", opts$num_positions, opts$num_rounds, exp_rows_counts, nrow(roi_counts)))
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
pos_and_frame <- lapply(roi_counts$filename, parsePositionAndFrame)
## Finalise the table by adding the position and frame information.
roi_counts$position <- sapply(pos_and_frame, function(e) e[["position"]])
roi_counts$frame <- sapply(pos_and_frame, function(e) e[["frame"]])
roi_counts[, .(position, frame, filename, count)] # more natural/intuitive sequence of columns
setKeyPF(roi_counts)

# Create the filtered counts table.
## First, do the actual data parsing...
infiles <- Sys.glob(file.path(opts$input_folder, pattern))
message(sprintf("Counting QC-passing bead ROIs in each of %s input files...", length(infiles)))
roi_counts_filtered <- data.table(do.call(what = rbind, args = lapply(infiles, countPassingQC)))
message("...done.")
message("Printing filtered counts table (before sorting)...")
roi_counts_filtered
## ...then, convert the values by column, according to column type...
type_by_column = list(position = as.integer, frame = as.integer, filename = as.character, count = as.integer)
for (col in names(type_by_column)) {
    convert = type_by_column[[col]]
    set(roi_counts_filtered, j = col, value = convert(roi_counts_filtered[[col]]))
}
## ...finally, set the key to prep the merge.
setKeyPF(roi_counts_filtered)

count_column_suffixes <- c("_unfiltered", "_filtered")

data_table_merged <- merge(
    roi_counts[, .(position, frame, count)], 
    roi_counts_filtered[, .(position, frame, count)], 
    by = c("position", "frame"), 
    suffixes = count_column_suffixes
)

# Make sure that NA becomes 0 in the counts columns.
for (count_suffix in count_column_suffixes) {
    count_colname <- sprintf("count%s", count_suffix)
    set(data_table_merged, which(is.na(data_table_merged[[count_colname]])), count_colname, 0)
}

data_file <- writeDataFile(data_table_merged)
message("Wrote combined data: ", data_file)

message("Building (frame, FOV) bead ROI count heatmaps")
roi_counts_heatmap <- buildCountsHeatmap(data_table_merged, "unfiltered")
saveCountsPlot(fig = roi_counts_heatmap, plot_type_name = "unfiltered.heatmap")
roi_counts_heatmap <- buildCountsHeatmap(data_table_merged, "filtered")
saveCountsPlot(fig = roi_counts_heatmap, plot_type_name = "filtered.heatmap")

message("Done!")
