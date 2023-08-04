# Import all the necessary packages.
library("argparse")
library("data.table")
library("ggplot2")
library("reshape2")

# Define some key constants.
## Measurement names and other column names
Z_COL <- "z_dc_rel"
Y_COL <- "y_dc_rel"
X_COL <- "x_dc_rel"
EUC_DIST_COL <- "euc_dc_rel"
DISTANCE_COLUMNS <- c(Z_COL, Y_COL, X_COL, EUC_DIST_COL)
NON_DISTANCE_COLUMNS <- c("full_pos", "t", "c", "QC")
COLUMNS_OF_INTEREST <- c(NON_DISTANCE_COLUMNS, DISTANCE_COLUMNS)
PLOT_RELEVANT_COLUMNS <- c("t", DISTANCE_COLUMNS)
## Output-related
OUTPUT_FILENAME_PREFIX <- "drift_correction_error"
COLORS4 <- c("darkmagenta", "cornflowerblue", "seagreen", "coral")
DISTANCE_LABELS <- c("z", "y", "x", "eucl")
names(DISTANCE_LABELS) <- c(Z_COL, Y_COL, X_COL, EUC_DIST_COL)
SCALE_FILL_4 <- scale_fill_manual(values = COLORS4, labels = DISTANCE_LABELS)
SCALE_COLOUR_4 <- scale_colour_manual(values = COLORS4, labels = DISTANCE_LABELS)

# Define the CLI and parse arguments.
cli_parser <- ArgumentParser(description="Visualise results of drift_correct_analsis.py, namely the amount of imprecision that remains even after drift correction")
cli_parser$add_argument("-i", "--fits-file", required=TRUE, help="Path to the fits file created by drift_correct_analysis.py")
cli_parser$add_argument("-o", "--output-folder", required=TRUE, help="Path to folder in which to place output files")
cli_parser$add_argument("--beads-channel", required=TRUE, type="integer", help="(0-based) channel in which beads were imaged")
cli_parser$add_argument("--handle-extant-output", default="OVERWRITE", choices=c("OVERWRITE", "overwrite", "SKIP", "skip", "FAIL", "fail"), help="Specification of how to handle case in which output path already exists")
opts <- cli_parser$parse_args()

# Read and filter the data.
message("Reading fits file: ", opts$fits_file)
if (!file_test("-f", opts$fits_file)) {
    stop("Missing fits file: ", opts$fits_file)
}
fits_data <- fread(opts$fits_file)[, ..COLUMNS_OF_INTEREST]
message("Filtering data for beads channels and for QC pass...")
fits_beads_pass_qc <- fits_data[QC == 1 & c == opts$beads_channel, ]
message(sprintf("After filtering for QC pass and beads channel, %s of %s records remain.", nrow(fits_beads_pass_qc), nrow(fits_data)))

# Define function to create data table for each round of plotting.
build_plot_table <- function(X, fov = NULL) {
    if (!is.null(fov)) {
        X <- X[full_pos == fov, ]
    }
    X <- data.table(melt(X[, ..PLOT_RELEVANT_COLUMNS], id.vars = "t"))
    X[, .(
        iqr_lower = quantile(value, 0.25), 
        median_value = quantile(value, 0.5), 
        iqr_upper = quantile(value, 0.75), 
        ci95_lower = mean(value) - 1.96 * sd(value) / sqrt(.N), 
        mean_value = mean(value), 
        ci95_upper = mean(value) + 1.96 * sd(value) / sqrt(.N)
    ), by = list(t, variable)]
}

# Plot means, with 95% CI flanking.
plot_means_and_intervals <- function(full_plot_data, fov = NULL) {
    X <- build_plot_table(full_plot_data, fov = fov)
    p <- ggplot(X, aes(x = t, y = mean_value, colour = variable, fill = variable)) + 
        geom_ribbon(aes(ymin = X$ci95_lower, ymax = X$ci95_upper), alpha=0.25) +
        geom_line(aes(y = X$mean_value), linewidth = 0.8) + 
        ylab("means and 95% CIs") + 
        geom_point(alpha = 0.5) + xlab("frame index") + theme_minimal() + 
        SCALE_FILL_4 + SCALE_COLOUR_4
    return(p)
}

## Plot medians, with IQR flanking.
plot_medians_and_iqrs <- function(full_plot_data, fov = NULL) {
    X <- build_plot_table(full_plot_data, fov = fov)
    p <- ggplot(X, aes(x = t, y = median_value, colour = variable, fill = variable)) + 
        geom_ribbon(aes(ymin = X$iqr_lower, ymax = X$iqr_upper), alpha=0.25) +
        geom_line(aes(y = X$median_value), linewidth = 0.8) + 
        ylab("medians and IQRs") + 
        geom_point(alpha = 0.5) + xlab("frame index") + theme_minimal() + 
        SCALE_FILL_4 + SCALE_COLOUR_4
    return(p)
}

# Generic function to plot lines connecting measures of center and shaded margins of interval / IQR.
plot_drift_correction_error <- function(full_plot_data, central_measure, fov) {
    if (is.null(fov)) {
        filt <- identity
        fov_text <- "all FOVs"
    } else {
        filt <- function(X) { X[full_pos == fov, ] }
        fov_text <- sprintf("FOV %s", fov)
    }
    if (central_measure == "mean") {
        plot_func <- plot_means_and_intervals
        method_text <- "Means and 95% intervals"
    } else if (central_measure == "median") {
        plot_func <- plot_medians_and_iqrs
        method_text <- "Medians and IQRs"
    } else {
        stop("Unkown method for plotting center of distribution: ", central_measure)
    }
    message("Creating plot based on: ", fov_text)
    title_text <- sprintf("%s, %s", method_text, fov_text)
    p <- plot_func(full_plot_data = full_plot_data, fov = fov) + 
        ggtitle(title_text) + 
        ylim(c(0, 100))
    return(p)
}

get_output_filepath <- function(fn) { file.path(opts$output_folder, fn) }

# Plot all data together.
plt_all_mean <- plot_drift_correction_error(full_plot_data = fits_beads_pass_qc, central_measure = "mean", fov = NULL)
all_data_means_filepath <- get_output_filepath(sprintf("%s.means.all_FOV.png", OUTPUT_FILENAME_PREFIX))
message("Saving all-means plot: ", all_data_means_filepath)
ggsave(filename = all_data_means_filepath, plot = plt_all_mean, bg = "white")
plt_all_median <- plot_drift_correction_error(full_plot_data = fits_beads_pass_qc, central_measure = "median", fov = NULL)
all_data_medians_filepath <- get_output_filepath(sprintf("%s.medians.all_FOV.png", OUTPUT_FILENAME_PREFIX))
message("Saving all-medians plot: ", all_data_medians_filepath)
ggsave(filename = all_data_medians_filepath, plot = plt_all_median, bg = "white")

# Save the per-FOV plots to files by method.
all_fovs <- unique(fits_beads_pass_qc$full_pos)
## First, the medians.
medians_plots_file <-  get_output_filepath(sprintf("%s.medians.by_FOV.pdf", OUTPUT_FILENAME_PREFIX))
message("Saving median plots by FOV: ", medians_plots_file)
pdf(file = medians_plots_file)
for (fov in all_fovs) {
    p <- plot_drift_correction_error(full_plot_data = fits_beads_pass_qc, central_measure = "median", fov = fov)
    print(p)
}
dev.off()
## Then, the means.
means_plots_file <-  get_output_filepath(sprintf("%s.means.by_FOV.pdf", OUTPUT_FILENAME_PREFIX))
message("Saving mean plots by FOV: ", means_plots_file)
pdf(file = means_plots_file)
for (fov in all_fovs) {
    p <- plot_drift_correction_error(full_plot_data = fits_beads_pass_qc, central_measure = "mean", fov = fov)
    print(p)
}
dev.off()
