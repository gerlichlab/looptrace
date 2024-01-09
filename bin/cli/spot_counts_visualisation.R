# Plot counts of detected regional barcode spots, aggregated by timepoint (probe/frame).

library("argparse")
library("data.table")
library("ggplot2")

kLocusSpecificName <- "locus-specific"
kPlotColor <- c("lightcoral", "indianred4")
kRegionalName <- "regional"
kSpotCountPrefix <- "spot_counts"

# Save a bead ROIs counts plot in the current output folder (by CLI), with a particular prefix for the name to indicate content and type.
#
# Args:
#   fig: the plot to save to disk
#   filtered_or_not: either 'filtered' or 'unfiltered', indicating whether the spots had been filtered
#   spot_type_name: either 'regional' or 'locus_specific', indicating the type of spots data used
saveCountsPlot <- function(fig, spot_type_name, plot_type_name, output_folder) {
    fn <- sprintf("%s.%s.%s.png", kSpotCountPrefix, spot_type_name, plot_type_name)
    plotfile <- file.path(output_folder, fn)
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
#
# Args:
#   spots_table: data table with (regional or locus-specific) spots data
#   filtered_or_not: either 'filtered' or 'unfiltered', indicating whether the spots have been filtered
#   spot_type_name: either 'regional' or 'locus_specific', indicating the type of spots data passed
plotAndWriteToFile <- function(spots_table, filtered_or_not, spot_type_name, output_folder) {
    if ( ! (filtered_or_not %in% c("filtered", "unfiltered"))) {
        stop("Illegal value for filtered/not argument: ", filtered_or_not)
    }
    p <- buildCountsHeatmap(spots_table)
    p <- p + ggtitle(sprintf("Counts of %s spots, %s", kRegionalName, filtered_or_not))
    saveCountsPlot(
        fig = p, 
        spot_type_name = spot_type_name, 
        plot_type_name = sprintf("%s.heatmap", filtered_or_not), 
        output_folder = output_folder
        )
}

############################################################################################
# Run program
############################################################################################
# CLI definition and parsing
## Parser
cli_parser <- ArgumentParser(description="Visualise bead ROI detection results")
## Required
cli_parser$add_argument("--filtered-spots-file", required = TRUE, help = "Path to filtered detected spots file")
cli_parser$add_argument("--unfiltered-spots-file", help = "Path to unfiltered detected spots file; required for regional spots")
cli_parser$add_argument("-o", "--output-folder", required = TRUE, help = "Path to folder in which to place output files")
cli_parser$add_argument("--spot-file-type", choices = c(kRegionalName, kLocusSpecificName), help = "Which kind of spots files are being provided")
## Parse the CLI arguments.s
opts <- cli_parser$parse_args()

# Handle the different execution branches (spots either from regional barcode or locus-specific probes).
if (opts$spot_file_type == kRegionalName) {
    if (is.null(opts$unfiltered_spots_file)) {
        stop(sprintf("For plotting %s spot counts, a path to an unfiltered file is additionally required.", kRegionalName))
    }
    # First, read the data tables.
    message("Reading unfiltered spots table: ", opts$unfiltered_spots_file)
    unfilteredSpots <- fread(opts$unfiltered_spots_file)
    message("Reading filtered spots table: ", opts$filtered_spots_file)
    filteredSpots <- fread(opts$filtered_spots_file)
    # Then, build the charts and save them to disk.
    plotAndWrite <- function(spots_table, filtered_or_not) { 
        plotAndWriteToFile(
            spots_table = spots_table, 
            filtered_or_not = filtered_or_not, 
            spot_type_name = opts$spot_file_type, 
            output_folder = opts$output_folder
            )
    }
    message(sprintf("Creating and saving %s spot counts plots...", kRegionalName))
    plotAndWrite(spots_table = unfilteredSpots, filtered_or_not = "unfiltered")
    plotAndWrite(spots_table = filteredSpots, filtered_or_not = "filtered")
    # Create a side-by-side grouped barchart, with unfiltered count next to filtered count for each timepoint.
    unfilteredSpots$filter_status <- FALSE
    filteredSpots$filter_status <- TRUE
    spotCountsCombined <- rbindlist(list(unfilteredSpots, filteredSpots))[, .(count = .N), by = list(filter_status, frame)]
    combinedBarchart <- ggplot(spotCountsCombined, aes(x = as.factor(frame), y = count, fill = filter_status)) + 
        geom_bar(stat = "identity", position = "dodge") + 
        xlab("timepoint") + 
        ggtitle(sprintf("Spot counts, %s", kRegionalName)) + 
        scale_fill_manual(name = element_blank(), values = kPlotColor, labels = c("unfiltered", "filtered")) + 
        labs(fill = element_blank()) + 
        theme_bw()
    saveCountsPlot(
        fig = combinedBarchart, 
        spot_type_name = opts$spot_file_type, 
        plot_type_name = "bargraph", 
        output_folder = opts$output_folder
        )
} else if (opts$spot_file_type == kLocusSpecificName) {
    message("Reading filtered spots table: ", opts$filtered_spots_file)
    filteredSpots <- fread(opts$filtered_spots_file)
    message(sprintf("Creating and saving %s spot counts plot...", kLocusSpecificName))
    plotAndWriteToFile(
        spots_table = filteredSpots, 
        filtered_or_not = "filtered", 
        spot_type_name = opts$spot_file_type, 
        output_folder = opts$output_folder
        )
    bargraph <- ggplot(filteredSpots[, .(count = .N), by = frame], aes(x = as.factor(frame), y = count)) + 
        geom_bar(stat = "identity") + 
        xlab("timepoint") + 
        ggtitle(sprintf("Spot counts, %s", kLocusSpecificName)) + 
        theme_bw()
    saveCountsPlot(
        fig = bargraph, 
        spot_type_name = opts$spot_file_type, 
        plot_type_name = "bargraph", 
        output_folder = opts$output_folder
        )
} else {
    stop("Illegal value for spot file type: ", opts$spot_file_type)
}

message("Done!")
