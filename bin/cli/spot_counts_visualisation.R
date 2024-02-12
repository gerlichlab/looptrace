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
    if ("frame_name" %in% colnames(spots_table)) { stop("Table already contains column for probe names!") }
    probe_index <- (spots_table$frame + 1)
    if ( ! all(probe_index %in% (1:length(probe_names))) ) {
        stop(sprintf(
            "Can't index (1-based, inclusive) into vector of %s probe names with these values: %s", 
            length(probe_names), 
            paste0(Filter(function(i) !(i %in% (1:length(probe_names))) , unique(probe_index)), collapse = ", ")
            ))
    }
    spots_table$frame_name <- sapply(probe_index, function(i) probe_names[i])
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
#   filt_type: string indicating type of spot filtration done
#   spot_type_name: either 'regional' or 'locus_specific', indicating the type of spots data passed
#   output_folder: path to folder in which to place output
plotAndWriteToFile <- function(spots_table, filt_type, spot_type_name, output_folder) {
    if (spot_type_name == kRegionalName) {
        legal_names <- c("unfiltered", "proximity-filtered", "nuclei-filtered")
    } else if (spot_type_name == kLocusSpecificName) {
        legal_names <- c("unfiltered", "filtered")
    } else {
        stop("Illegal value for spot_type_name: ", spot_type_name)
    }
    if ( ! (filt_type %in% legal_names)) {
        stop(sprintf(
            "For spot_type_name %s, illegal value for filtered/not argument: %s", 
            spot_type_name, 
            filt_type
            ))
    }
    p <- buildCountsHeatmap(spots_table)
    p <- p + ggtitle(sprintf("Counts of %s spots, %s", spot_type_name, filt_type))
    p <- p + theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
    saveCountsPlot(
        fig = p, 
        spot_type_name = spot_type_name, 
        plot_type_name = sprintf("%s.heatmap", filt_type), 
        output_folder = output_folder
        )
}

# Save a bead ROIs counts plot in the current output folder (by CLI), with a particular prefix for the name to indicate content and type.
#
# Args:
#   fig: the plot to save to disk
#   spot_type_name: either 'regional' or 'locus_specific', indicating the type of spots data used
#   plot_type_name: name for the type of chart to produce, e.g. 'barchart', used to build filename
#   output_folder: path to folder in which to place output
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
    help = "Path to filtered detected spots file, main input data file"
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
    "--nuclei-filtered-spots-file", 
    help = "Path to detected spots file after applying proximity AND nuclei filtration; only used for regional spots"
)
cli_parser$add_argument(
    "--unfiltered-spots-file", 
    help = "Path to unfiltered detected spots file; required iff spots data is regional; only used for regional spots"
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
    readData <- function(filt_type, spot_file) {
        message(sprintf("Reading table and adding probe names for %s %s spots table: %s", filt_type, kRegionalName, spot_file))
        addProbeNames(fread(spot_file), opts$probe_names)
    }
    plotAndWrite <- function(spots_table, filt_type) { 
        plotAndWriteToFile(
            spots_table = spots_table, 
            filt_type = filt_type, 
            spot_type_name = opts$spot_file_type, 
            output_folder = opts$output_folder
            )
    }
    spots_table_files <- list(
        proximity_filtered = opts$filtered_spots_file, 
        unfiltered = opts$unfiltered_spots_file
        )
    if (!is.null(opts$nuclei_filtered_spots_file)) {
        spots_table_files$nuclei_filtered <- opts$nuclei_filtered_spots_file
    }
    # First, read the data tables.
    spot_tables <- lapply(names(spots_table_files), function(filt_type) readData(filt_type = filt_type, spot_file = spots_table_files[[filt_type]]))
    message("Names of spot tables: ", paste0(names(spot_tables), collapse = ", "))
    # Then, build the charts and save them to disk.
    message(sprintf("Creating and saving %s spot counts plots...", kRegionalName))
    plotfiles <- lapply(names(spot_tables), function(filt_type) plotAndWrite(spots_table = spot_tables[[filt_type]], filt_type = filt_type))
    message("Saved plot files: ", paste0(plotfiles, collapse = ", "))

    # Create a side-by-side grouped barchart, with unfiltered count next to filtered count for each timepoint.
    spotCountsCombined <- rbindlist(lapply(names(spot_tables), function(filt_type) {
        dat_tab <- spot_tables[[filt_type]]
        dat_tab[, .(frame_name = frame_name, filter_status = filt_type)]
    }))[, .(count = .N), by = list(filter_status, frame_name)]
    combinedBarchart <- ggplot(spotCountsCombined, aes(x = as.factor(frame_name), y = count, fill = filter_status)) + 
        geom_bar(stat = "identity", position = "dodge") + 
        xlab("probe") + 
        ggtitle(sprintf("Spot counts, %s", kRegionalName)) + 
        scale_fill_manual(name = element_blank(), values = kPlotColor, labels = names(spot_tables)) + 
        labs(fill = element_blank()) + 
        theme_bw()
    saveCountsPlot(
        fig = combinedBarchart, 
        spot_type_name = opts$spot_file_type, 
        plot_type_name = "bargraph", 
        output_folder = opts$output_folder
        )
} else if (opts$spot_file_type == kLocusSpecificName) {
    if (!is.null(opts$unfiltered_spots_file)) { warning(sprintf(
        "Unfiltered spots file was provided (%s), but it's not used for %s spot counts visualisation"
    )) }
    message("Reading filtered spots table: ", opts$filtered_spots_file)
    filteredSpots <- fread(opts$filtered_spots_file)
    message(sprintf("Creating and saving %s spot counts plot...", kLocusSpecificName))
    plotAndWriteToFile(
        spots_table = filteredSpots, 
        filt_type = "filtered", 
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
