# Plot counts of detected regional barcode spots, aggregated by timepoint (probe/frame).

library("argparse")
library("data.table")
library("ggplot2")

kLocusSpecificName <- "locus-specific"
kPlotColor <- c("lightcoral", "indianred4")
kRegionalName <- "regional"
kSpotCountPrefix <- "spot_counts"

# Augment the (regional) spots table with probe names, since--unlike for the 
# locus-specific spots table--that information's not added to the regional spots table.
#
# Args:
#   spots_table: data table with regional spots data
#   probe_names: ordered list of names to give the imaging timepoints
addProbeNames <- function(spots_table, probe_names) {
    if ("frame_name" %in% colnames(spots_table)) { stop("Table alread contains column for probe names!") }
    spots_table$frame_name <- sapply(spots_table$frame, function(i) probe_names[i + 1])
    spots_table
}

# Create the heatmap for the given counts data (1 count per (FOV, time) pair).
buildCountsHeatmap <- function(spots_table) {
    counts_table <- spots_table[, .(count = .N), by = list(position, frame_name)]
    ggplot(counts_table, aes(x = frame_name, y = position, fill = count)) + 
        geom_tile() + 
        xlab("probe") + 
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
    p <- p + theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
    saveCountsPlot(
        fig = p, 
        spot_type_name = spot_type_name, 
        plot_type_name = sprintf("%s.heatmap", filtered_or_not), 
        output_folder = output_folder
        )
}

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

############################################################################################
# Run program
############################################################################################
# CLI definition and parsing
## Parser
cli_parser <- ArgumentParser(description="Visualise bead ROI detection results")
## Required
cli_parser$add_argument(
    "--filtered-spots-file", 
    required = TRUE, 
    help = "Path to filtered detected spots file"
    )
cli_parser$add_argument(
    "--probe-names", 
    nargs = "*",
    help = "Ordered list of names, used to each frame/timepoint with a probe; required iff spots data is regional"
    )
cli_parser$add_argument(
    "-o", "--output-folder", 
    required = TRUE, 
    help = "Path to folder in which to place output files"
    )
cli_parser$add_argument(
    "--unfiltered-spots-file", 
    help = "Path to unfiltered detected spots file; required iff spots data is regional"
    )
cli_parser$add_argument(
    "--spot-file-type", 
    choices = c(kRegionalName, kLocusSpecificName), 
    help = "Which kind of spots files are being provided"
    )
## Parse the CLI arguments.s
opts <- cli_parser$parse_args()

# Handle the different execution branches (spots either from regional barcode or locus-specific probes).
if (opts$spot_file_type == kRegionalName) {
    if (is.null(opts$unfiltered_spots_file) || is.null(opts$probe_names)) {
        stop(sprintf(
            "For plotting %s spot counts, a list of probe names and a path to an unfiltered file are additionally required.", 
            kRegionalName
            ))
    }
    addProbes <- function(spots_table) { addProbeNames(spots_table, opts$probe_names) }
    # First, read the data tables.
    message("Reading unfiltered spots table: ", opts$unfiltered_spots_file)
    unfilteredSpots <- addProbes(fread(opts$unfiltered_spots_file))
    message("Reading filtered spots table: ", opts$filtered_spots_file)
    filteredSpots <- addProbes(fread(opts$filtered_spots_file))
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
    spotCountsCombined <- rbindlist(list(
        unfilteredSpots[, .(filter_status, frame_name)], 
        filteredSpots[, .(filter_status, frame_name)]
    ))[, .(count = .N), by = list(filter_status, frame_name)]
    combinedBarchart <- ggplot(spotCountsCombined, aes(x = as.factor(frame_name), y = count, fill = filter_status)) + 
        geom_bar(stat = "identity", position = "dodge") + 
        xlab("probe") + 
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
    bargraph <- ggplot(filteredSpots[, .(count = .N), by = frame_name], aes(x = frame_name, y = count)) + 
        geom_bar(stat = "identity") + 
        xlab("probe") + 
        ggtitle(sprintf("Spot counts, %s", kLocusSpecificName)) + 
        theme_bw() + 
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
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
