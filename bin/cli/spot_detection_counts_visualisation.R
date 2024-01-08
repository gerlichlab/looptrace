# Plot counts of detected regional barcode spots, aggregated by timepoint (probe/frame).

library("argparse")
library("data.table")
library("ggplot2")

# CLI definition and parsing
## Parser
cli_parser <- ArgumentParser(description="Visualise bead ROI detection results")
## Required
cli_parser$add_argument("--unfiltered-spots-file", required = TRUE, help = "Path to unfiltered detected spots file")
cli_parser$add_argument("--filtered-spots-file", required = TRUE, help = "Path to filtered detected spots file")
cli_parser$add_argument("-o", "--output-folder", required = TRUE, help = "Path to folder in which to place output files")
## Parse the CLI arguments.s
opts <- cli_parser$parse_args()

kSpotCountPrefix <- "spot_counts"
kPlotTitlePrefix <- "Counts of detected spots"
kPlotColor <- c("lightcoral", "indianred4")

# Save a bead ROIs counts plot in the current output folder (by CLI), with a particular prefix for the name to indicate content and type.
saveCountsPlot <- function(fig, plot_type_name) {
    plotfile <- file.path(opts$output_folder, sprintf("%s.%s.png", kSpotCountPrefix, plot_type_name))
    message("Saving plot: ", plotfile)
    ggsave(filename = plotfile, plot = fig)
    plotfile
}

# Create the heatmap for the given counts data (1 count per (FOV, time) pair).
buildCountsHeatmap <- function(spots_table) {
    counts_table <- spots_table[, .(count = .N), by = list(position, frame)]
    counts_table$frame <- as.factor(counts_table$frame)
    ggplot(counts_table, aes(x = frame, y = position, fill = count)) + 
        geom_tile() + 
        xlab("timepoint") + 
        theme_bw()
}

# Create a heatmap and save the plot to an appropriately named file in the current output folder.
plotAndWriteToFile <- function(spots_table, filtered_or_not) {
    if ( ! (filtered_or_not %in% c("filtered", "unfiltered"))) {
        stop("Illegal value for filtered/not argument: ", filtered_or_not)
    }
    p <- buildCountsHeatmap(spots_table)
    p <- p + ggtitle(sprintf("%s, %s", kPlotTitlePrefix, filtered_or_not))
    saveCountsPlot(fig = p, plot_type_name = sprintf("%s.heatmap", filtered_or_not))
}

############################################################################################
# Run program
############################################################################################

# First, read the data tables.
message("Reading unfiltered spots table: ", opts$unfiltered_spots_file)
unfilteredSpots <- fread(opts$unfiltered_spots_file)
message("Reading filtered spots table: ", opts$filtered_spots_file)
filteredSpots <- fread(opts$filtered_spots_file)

# Then, build the charts and save them to disk.
message("Creating and saving plots...")
plotAndWriteToFile(spots_table = unfilteredSpots, filtered_or_not = "unfiltered")
plotAndWriteToFile(spots_table = filteredSpots, filtered_or_not = "filtered")

# Create a side-by-side grouped barchart, with unfiltered count next to filtered count for each timepoint.
unfilteredSpots$filter_status <- FALSE
filteredSpots$filter_status <- TRUE
spotCountsCombined <- rbindlist(list(unfilteredSpots, filteredSpots))[, .(count = .N), by = list(filter_status, frame)]
combinedBarchart <- ggplot(spotCountsCombined, aes(x = as.factor(frame), y = count, fill = filter_status)) + 
    geom_bar(stat = "identity", position = "dodge") + 
    xlab("timepoint") + 
    ggtitle(kPlotTitlePrefix) + 
    scale_fill_manual(name = element_blank(), values = kPlotColor, labels = c("unfiltered", "filtered")) + 
    labs(fill = element_blank()) + 
    theme_bw()
saveCountsPlot(fig = combinedBarchart, plot_type_name = "combined.bargraph")

message("Done!")
