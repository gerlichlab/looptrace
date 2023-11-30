library("argparse")
library("data.table")
library("ggplot2")

cli_parser <- argparse$ArgumentParser(description="Visualise properties about aggregates of intensities of spot images used for tracing")
cli_parser$add_argument("-i", "--infile", required=TRUE, help="Path to input file, i.e. produced by related Python functions/program")
cli_parser$add_argument("-O", "--outfolder", required=TRUE, help="Path to folder in which to place output files")
cli_parser$add_argument("--regionalFrames", required=TRUE, nargs="+", help="Indices (0-based) of regional barcode timepoints/frames")

opts <- cli_parser$parse_args()
if (!file_test("-f", opts$infile)) { stop("Missing input file for trace spots images analysis: ", opts$infile) }
message("Regional barcode frames: ", paste0(opts$regionalFrames, collapse = ", "))

message("Reading input file: ", opts$infile)
tst <- fread(opts$infile)

# Augment the table with labels about which frames are regional barcodes, and with aggregate-of-aggregate stats.
tst[, `:=`(is_regional = (frame %in% opts$regionalFrames), detection_source = (frame == ref_frame))]
tst[, `:=`(max_max = max(max), max_mean = max(mean), max_median = max(median)), by = roi_id]

# See the distribution of percent-of-max values by regional frame.
# Percent-of-max is computed for each ROI, as the 100 * max_in_ref/max_all, 
# where max_in_ref is the max intensity in the ROI image corresponding to the frame 
# in which the ROI was detected, and max_all is the max intensity across all frames 
# within that particular ROI. This plot then groups these values by ref_frame.
ggplot(tst[frame == ref_frame, .(ref_frame, pct_of_max=100*(max/max_max))], aes(x=as.factor(ref_frame), y=pct_of_max, fill=ref_frame)) + 
    geom_violin() + 
    xlab("Regional barcode") + 
    ylab("Percent of max") + 
    theme_bw() +
    labs(fill = "region") + 
    ggtitle("Regional barcode's maximum's percent of maximum across all frames per ROI")

ggplot(tst[frame == ref_frame, .(pct_zero_max=sum(max==0)/.N), by=ref_frame], aes(x=as.factor(ref_frame), y=pct_zero_max)) + 
    geom_col() + 
    xlab("Regional barcode") + 
    ylab("Percent with max = 0") + 
    theme_bw() + 
    ggtitle("Percentage of ROIs with 0 max in frame of ROI detection")

classification_of_max_mean <- tst[, .(correct_region = any((frame==ref_frame) & (mean==max_mean)), wrong_region = any((frame != ref_frame) & is_regional & (mean==max_mean)), non_region = any(!is_regional & (mean==max_mean))), by = roi_id]
max_mean_classes_wide <- merge(tst[frame == ref_frame, ], classification_of_max_mean, by = "roi_id")
max_mean_classes_long <- melt(max_mean_classes_wide, id.vars = c("ref_frame"), measure.vars = c("correct_region", "wrong_region", "non_region"))
ggplot(max_mean_classes_long[, .(pct = sum(value)/.N), by=list(ref_frame, variable)], aes(x=as.factor(ref_frame), y=pct, fill=variable)) + 
    geom_col(position="dodge") + 
    xlab("Regional barcode") + 
    ylab("proportion") + 
    theme_bw() + 
    labs(fill = "max mean type") +
    ggtitle("Proportion of ROIs by categorisation of frame of greatest average intensity")
