# `looptrace` parameters configuration file
The parameters configuration file is where you tell `looptrace` about various algorithmic and data analysis settings to use, as well as where to read and write data.

## Requirements and suggestions
* Check that each channel setting (often with a `_ch` or `_channel` suffix) matches what's been used in the imaging experiment which generated the data to be processed.
* `xy_nm` and `z_nm` should be adjusted to match your microscope settings (number of nanomerters per step in xy or in z).
* `nuc_method` should be set to "cellpose".
* `nuc_3d` should be absent or set to `False`.
* `analysis_path` should be an _absolute_ path but may use environment and/or user variables.
* `analysis_path` should specify the path to a folder that exists before the pipeline is run.
* `zarr_conversions` should be a mapping from subfolder in the images folder (`--images-folder` when [running from the command-line](./running-the-pipeline.md#general-workflow)) to new subfolder (1-to-1): the keys are names of subfolders with raw image files (e.g., `.nd2`), and each value will be the new folder with that image data, just reformatted as `.zarr`. 
Typically there will be one entry for the sequential FISH images' folder and another for the nuclei images' folder.
* Check that `spot_wavelength` and `objective_na` have been adjusted to match the microscope and fluorophores used.
* `decon_psf` should be set to `gen`.
* `require_gpu` should be set to `True`.
* `decon_iter` should be set to a nonnegative integer. If you don't want to run deconvolution, set this to 0. Setting this to 0 obviates the need for NVIDIA and GPUs, so you could change `nvidia-docker` to simply `docker` when [running the pipeline](./running-the-pipeline.md#general-workflow).
* `decon_input_name` should likely be the single value in the `zarr_conversions` mapping.
* `reg_input_template` and `reg_input_moving` should likely match each other and should correspond to adding a `_decon` suffix to the value of `decon_input_name`.
* `reg_ref_timepoint` should be set to something approximately equal to the middle of the imaging timecourse (i.e, midway between first and last timepoint).
* `num_bead_rois_for_drift_correction` should be set to 100 or 200 (number of beads to use for drift correction).
Having bead count fewer than the value will impede processing, but higher values _may_ given a bit better drift correction.
Judge in accordance with how many beads you anticipate having per image.
* `num_bead_rois_for_drift_correction_accuracy` should be set to 100.
* `coarse_drift_downsampling` should be set to 2; use 1 for no downsampling.
* `detection_method` should be set to `dog`, and `spot_threshold` to 15. If using `intensity`, a much higher `spot_threshold` will be needed.
* `distanceBeneathWhichSpotRoisWillMerge` should be set to a positive value if you want to do mergers of regional barcode spot ROIs which are close together. If you don't wish to do that merger, omit this configuration key. The value represents a Euclidean distance and must encode the value itself along with the units, e.g. `1500 nm`.
* `parallelise_spot_detection` should be set to `False`.
* `spot_downsample` should be a small integer, often just 2 or even 1 (no downsampling).
* `spot_in_nuc` should be set to `True`, generally.
* `padding_method` should be set to `edge`.
* `tracing_cores` should be a value no more than the number of CPUs on the machine on which the processing will run.
* `mask_fits` should be set to `False`.
* `roi_image_size` should be a 3-element list and should most likely be (8, 16, 16) or (16, 32, 32).
* `subtract_background` should be set to 0 (or whatever imaging timepoint is to be used as the reference for the notion of "only noise").
* Check that the tracing QC parameters are as desired:
    * For `A_to_BG`, 2 is often a good setting.
    * For `sigma_xy_max`, 150 is often a good setting.
    * For `sigma_z_max`, 400 is often a good setting.
    * For `max_dist`, 800 nm is often a good setting.
* Note that `max_dist` must be a nonnegative value carrying units, e.g. `800 nm`.
* If you want the Numpy arrays representing the spot images for tracing (the `*.npy` files) to be kept even after zipping, set `keep_spot_images_folder` to `True`.

### Filtration of FISH spots by proximity to beads
* `proximityFiltrationBetweenBeadsAndSpots` must be set (for now) and should be either...
    * A Boolean, which for now must be `True`, implying that `subtract_background` must be used and correspond to an imaging timepoint / round. In the future this step will be skippable, and this key could be omitted or set to `False`.
    * A nonnegative integer corresponding to an imaging timepoint / round
* `beadSpotProximityDistanceThreshold` must be set to a physical units value of length in nanometers, e.g. `200 nm`.
This represents the minimum distance a FISH spot centroid must be from a bead centroid to not be discarded on suspicion of being actually a bead.
