# Pipeline outputs
This document makes reference to various folders for pipeline inputs and outputs. 
If you're not yet roughly familiar with how to run the pipeline and how it works, please first read [this document](./running-the-pipeline.md). 

Also referenced here are the names of various ___stages___ of the pipeline. 
If you're not familiar with these, please refere to the [relevant section](./pipeline-execution-control-and-rerun.md#pipeline-stage-names) of the document about [pipeline control flow](./pipeline-execution-control-and-rerun.md).

* `zarr_conversion`: writes to the images folder, a subfolder with ZARR versions of the input images, defined by the `zarr_conversions` section of the [parameters configuration file](./parameters-configuration-file.md).
* `nuclei_detection`: produces the nuclear masks subfolder stored in the images folder
* `nuclei_drift_correction`: produces `*_nuclei_drift_correction_coarse.csv`, storing shifts for nuclei relative to the reference timepoint in the experiment. This is stored in the analysis folder and used for adjusting the positions of the nuclear masks when filtering FISH spots for filtration in nuclei.
In general, you should not nead to look at this.
* `drift_correction__coarse`: produces something like `*_zarr_drift_correction_coarse.csv` (or similar), which is the coarse-grained shift of each imaging timepoint relative to a reference timepoint (defined in the [parameters configuration file](./parameters-configuration-file.md)). 
In general, you should not nead to look at this. 
This is used by the fine drift correction and by spot bounding, which determines coordinates of subarrays from which to extract pixel data for tracing.
* `bead_roi_generation`: produces the `bead_rois` subfolder of the analysis folder, in which initially there are files containing information about each detected bead ROI. In general, you should not need to look at these.
It's only the bead partitioning step which directly uses these files contained thereir.
* `bead_roi_detection_analysis`: produces the charts/graphs with counts of bead ROIs, grouped/grouped stratified different ways, written into the analysis folder. 
Use these to diagnose potential errors or odd behavior of drift correction.
* `bead_roi_partition`: produces two subfolders within the `bead_rois` subfolder in the analysis folder. These represent the isolation of subsets of fiducial beads for drift correction and then for drift correction accuracy analysis. In general, you should not need to look at these.
Thse are used by the fine drift correction and the drift correction accuracy analysis.
* `drift_correction__fine`: produces the `*_zarr_drift_correction_fine.csv` (or similar), which is used for the drift correction accuracy analysis and for the tracing.
* `drift_correction_accuracy_analysis`: produces the charts/graphics related to the assessment of what drift remains even after drift correction. 
This is something you'll often want to look at, and it uses the coarse and fine drift correction information, as well as the results of the bead partitioning.
* `spot_detection`: produces the `*_rois.csv` file representing all _regional_ barccode spots strong enough to be detected according to the [configuration file](./parameters-configuration-file.md). 
This should operate directly on the raw (though possibly deconvoluted) image data in `zarr`, and this must be properly configured in the [configuration file](./parameters-configuration-file.md).
* `spot_proximity_filtration`: produces the `*_rois.proximity_labeled.csv` and `*_rois.proximity_filtered.csv`. 
The labeled file has a column for which rows of the file are related as neighbors (per the configuration options in the [parameters config](./parameters-configuration-file.md) and the [imaging rounds config](./imaging-rounds-configuration-file.md)). 
This uses the output of the spot detection.
The filtered file is the same as the labeled, except for the exclusion of the rows in which the neighbors column was nonempty, and the subsequent dropping of that column.
* `spot_nucleus_filtration`: produces the `*.nuclei_labeled.csv` and `*.nuclei_filtered.csv` files, which represent association of a nucleus label to each spot (`0` representing that the spot's center isn't found within the border of a nuclear mask), and the filtration of the rows to which that applies. More or less analogous in this respect to the proximity-labeled and -filtered files.
This uses the output of the spot proximity filtration.
* `spot_counts_visualisation__regional`: produces the charts related to the counts of (_regional_ barcode) spots. This uses the output from the spot detection and filtration by proximity and nuclei.
* `spot_bounding`: this computes the drift-corrected bounding boxes from which to extract pixel data that will be used for tracing. This uses the filtered spot detection output (either just the proximity-filtered or proximity- and nuclei-filtered, depending on the `spot_in_nuc` setting of the [parameters configuration file](./parameters-configuration-file.md)), as well as the coarse and fine drift correction outputs. 
This produces the `*_dc_rois.csv` file.
* `spot_extraction`: this ues the output of the spot bounding step and produces the `spot_images_dir`, which may be retained only temporarily. Inside are numpy array files representing the volume extracted according to each the 3D bounding box, which is to what fits will be aplpied during tracing. 
You'll roughly never need to directly use the output of this step.
* `spot_zipping`: uses the output of spot extraction to produce the `spot_images.npz` file which contains the individual 3D numpy array files that are used as the locus spot images for tracing. 
You'll roughly never need to directly use the output of this step.
* `tracing`: this produces the `*_traces.csv` file by fitting a 3D Gaussian to each of the 3D volumes extracted from individual timepoints in each regional spot (in the spot extraction step).
This uses the output of spot zipping and coarse + fine drift correction.
You won't work directly with this step's output, but rather with that of its successors.
* `spot_region_distances` uses the output of the tracing and of the drift correction to compute the distance between the center of each locus-specific spot and its corresponding regional spot's center. 
It also maps timepoint to frame name. 
It produces the `*.traces.enriched.csv` file. 
Generally you won't work directly with this file, but perhaps one of the ones produced subsequently by the tracing QC step.
* `tracing_QC` produces the `*.traces.enriched.unfiltered.csv` and `*.traces.enriched.filtered.csv` file. 
It uses the `*.traces.enriched.csv` file from tracing the tracing /  spot region distances steps and the coarse + fine drift correction file produced from fine drift correction. 
These outputs can be useful for your own downstream analysis.
* `spot_counts_visualisation__locus_specific` produces the spot counts charts / graphics, using the outputs of the tracing QC step. This can be quite useful to inspect.
* `pairwise_distances__locus_specific` uses the imaging rounds config and the output of the tracing QC step to produce the `*.pairwise_distances__locus_specific.csv` file. 
This is often useful for subsequent analysis.
* `pairwise_distances__regional` uses the coordinates of the regional barcodes computed precisely (with fine drift correction) during tracing. 
However, the dependence on drift correction is only indirect, so if nothing about your pipeline restart affects the drift correction, you don't need to rerun it. 
The only direct dependence should be on the coordinates from the enriched, filtered traces file.
* `locus_specific_spots_visualisation_data_prep` uses the data from the `spot_images.npz` file to produce ZARR arrays which may then be dragged-and-dropped into `napari` (and likely some other programs) for visualisation. These arrays are stored in the `locus_spots_visualisation` subfolder of the analysis folder, along with a pair of CSV files for each field of view. Each FOV should have its own folder, containing the ZARR and the two CSVs.
This pair of files is then typically also dragged-and-dropped into `napari` for visualisation. 
For more about the use of these outputs, refer to the [visualisation doc](./visualisation.md).
* `nuclear_masks_visualisation_data_prep` produces data similar to the locus-specific spots visualisation step, aimed at facilitating visualistation of the nuclear regions used for spot filtration.
* `regional_spots_visualisation_data_prep` uses the 3 regional spots files produced by `spot_detection`, `spot_proximity_filtration`, and `spot_nucleus_filtration` to produce `regional_spots_visualisation`, with per-FOV data to drag-and-drop into Napari for visualisation.
