# This is a series of commands to be run interactively to visualise the results of 
# a collection of runs of spot detection over different parameterisations. 
# Paths and maybe other values will need to be updated for a new analysis.

library("data.table")
library("ggplot2")

IMBA_GREEN <- "chartreuse4"
IMBA_RED <- "firebrick1"

# Set up the file paths.
exp_home <- file.path("/mnt/005831")
gridsearch_home <- file.path(exp_home, "2023-06-20__spot_detection_runs")
gs_data_file <- file.path(gridsearch_home, "gridsearch2_results.csv")

# Confirm we're working with a real file, then read it.
stopifnot(file_test("-f", gs_data_file))
gsres <- fread(gs_data_file)

# Examine what we're working with.
colnames(gsres)

# Check the number of spots found when filtering for nuclei.
table(gsres[only_in_nuclei == "TRUE", spot_count])

# Split the data into null- and non-null-valued for spot count.
# Null valued corresponds to a parameterisation that took too long to run.
nulls <- gsres[is.na(spot_count), ]
gs_nn <- gsres[ ! is.na(spot_count), ]

# Count how many parameterisations of each spot detection method there are.
table(gs_nn[, method])
# Crudely see how nucleus inclusion filtration affects spot count.
table(gs_nn[, .(only_in_nuclei, spot_count > 0)])


# Plot some data for counts of spots detected.
gs_plottable <- gs_nn[only_in_nuclei == "FALSE", ]
gs_plottable$subtract_crosstalk <- factor(gs_plottable$subtract_crosstalk, levels=c("FALSE", "TRUE"), labels=c("no", "yes"))
COLORS__NO_YES <- c(no=IMBA_GREEN, yes=IMBA_RED)
## intensity-based
intensity_plot <- ggplot(gs_plottable[method == "intensity", ], aes(x=as.factor(threshold), y=spot_count, fill=subtract_crosstalk)) + 
  geom_bar(stat="identity", position="dodge") + 
  facet_wrap(. ~ frames) + 
  xlab("threshold") + 
  ylab("number of spots") + 
  scale_fill_manual("(-) crosstalk", values = COLORS__NO_YES) +
  ggtitle("Spot count for intensity-based detection parameterisations")
intensity_plot
intensity_based_outfile <- file.path(gridsearch_home, "spot_counts__intensity_based.png")
message("Saving intensity-based plot: ", intensity_based_outfile)
ggsave(filename=intensity_based_outfile, plot=intensity_plot)
## difference of Gaussians
### Discard the very-low-threshold DoG results, as they appear to have too many spots.
gaussian_plot <- ggplot(gs_plottable[(method == "dog" & threshold > 5), ], aes(x=as.factor(threshold), y=spot_count, fill=subtract_crosstalk)) + 
  geom_bar(stat="identity", position="dodge") + 
  facet_wrap(. ~ frames) + 
  xlab("threshold") + 
  ylab("number of spots") + 
  scale_fill_manual("(-) crosstalk", values = COLORS__NO_YES) +
  ggtitle("Spot count for DoG-based detection parameterisations")
gaussian_plot
dog_based_outfile <- file.path(gridsearch_home, "spot_counts__DoG_based.png")
message("Saving DoG-based plot: ", dog_based_outfile)
ggsave(filename=dog_based_outfile, plot=gaussian_plot)
