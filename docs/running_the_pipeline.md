# How to run the main `looptrace` processing pipeline
This document describes the essentials needed to run the main `looptrace` processing pipeline. This is primarily for end-users and does not describe much about the software design or packaging.


## Minimal requirements
To be able to run this pipeline on the lab machine, these are the basic requirements:
* __Have an account on the machine__: you should be able to authenticate with your normal username/password combination used for most authentication within the institute.
* __Be in the `docker` group__: If you've not run something with `docker` on this machine previously, you're most likely not in this group. Ask Vince or Chris to add you.


## Data layout and organisation
* __Main experiment folder__ (`CURR_EXP_HOME` environment variable): On the cluster and on the lab machine, this is often something like `/path/to/experiments/folder/Experiments_00XXXX/00XXXX`, but it could be anything so long as the substructure matches what's expected / defined in the config file.
* __Images subfolder__ (created on lab machine or cluster): something like `images_all`, but just needs to match the value you'll give with the `--images-folder` argument when running the pipeline.
    * _Raw nuclei images_ subfolder: something like `nuc_images_raw`, though just must match the corresponding key in the config file
    * _FISH images_ subfolder: something like `seq_images_raw`, though just must match the corresponding key in the config file
* __Pypiper subfolder__ (created on lab machine or on cluster): something like `pypiper_output`, where pipeline logs and checkpoints are written; this will be passed by you to the pipeline runner through the `-O / --output-folder` argument when you run the pipeline.
* __Analysis subfolder__ (created on lab machine or on cluster): something like `2023-08-10_Analysis01`, though can be anything and just must match the name of the subfolder you supply in the config file, as the leaf of the path in the `analysis_path` value


## _Imaging rounds_ configuration file
This is a file that declares the imaging rounds executed over the course of an experiment.
These data should take the form of a mapping, stored as a JSON object; the sections are described below.
An example is available in the [test data](../src/test/resources/TestImagingRoundsConfiguration/example_imaging_round_configuration.json).

* `regionGrouping` is required and should be a mapping, with at minimum a `semantic` key. The value for the semantic may be "Trivial", "Permissive", or "Prohibitive", depending on how you want the groupings to be interpreted with respect to excluding proximal spots. 
"Trivial" treats all regional spots as one big group and will exclude any that violate the proximity threshold; it's like "Prohibitive" but with no grouping.
"Permissive" allows spots from timepoints grouped together to violate the proximity threshold; spots from timepoints not grouped together may not violate the proximity threshold.
"Prohibitive" forbids spots from timepoints grouped together to violate the proximity threshold; spots from timepoints not grouped together may violate the proximity threshold.
* If `semantic` is set to "Trivial", there can be no `groups`; otherwise, `groups` is required and must be an array-of-arrays.
* `semantic` and `groups` must be nested within the `regionGrouping` mapping.
* If specified, the value for `groups` must satisfy these properties:
    * Each group must have no repeat value.
    * The groups must be disjoint.
    * The union of the groups must cover the set of regional round timepoints from the `imagingRounds`.
    * The set of regional round timepoints from the `imagingRounds` must cover the union of groups.
* `minimumPixelLikeSeparation` must be a key in the `regionGrouping` mapping and should be a nonnegative integer. 
This represents the minimum separation (in pixels) in each dimension between the centroids of regional spots. 
NB: For z, this is slices not pixels.
* `imagingRounds` must be present, and it enumerates the imaging rounds that comprise the experiment. 
For every entry, `time` is required (as a nonnegative integer), and either `probe` or `name` is required.
No `time` value can be repeated, and the timepoints should form the sequence (0, 1, 2, ..., *N*-1), where *N* is the number of imaging rounds in the experiment.
Each value in this array represents a single imaging round; each round is one of 3 cases:
    * _Blank_ round: specify `time` as a nonnegative integer, and `name` as a string; set `isBlank` to `true`.
    * _Locus_ FISH round: specify `time` as a nonnegative integer, omit `isBlank` or set it to `false`, and specify `probe` as a string. If this is a repeat of some earlier timepoint, specify `repeat` as a positive integer. `name` can be specified or inferred from `probe` and `repeat`; `isRegional` must be absent or `false`.
    * _Regional_ FISH round: this is as for a locus-specific round, but ensure that `isRegional` is set to `true`, and this cannot have a `repeat` (omit it.)
* For any non-regional round, `repeat` may be included with a natural number as the value. In that case, if `name` is absent (derived from `probe`), the suffix `_repeatX` will be added to the `probe` value to determine the name.
* In general, if `name` is absent, it's set to `probe` value (though this is not possible for a blank round, which is why `name` is required when `isBlank` is `true`).
* The set of names for the rounds of the imaging sequence must have no repeated value (including a collision of a derived value with some explicit or derived value).
* `locusGrouping` must be present as a top-level key if and only if there's one or more locus imaging rounds in the `imagingRounds`.
    * Each key must be a string, but specifying the timepoint of one of the _regional_ rounds from the `imagingRounds` sequence.
    * Each value is a list of locus imaging timepoints associated with the regional timepoint to which it's keyed.
    * If present, the values in the lists must be such that the union of the lists is the set of locus imaging timepoints from the `imagingRounds` sequence.
    * The values lists should have unique values, and they should be disjoint.
* Check that the list of `illegal_frames_for_trace_support` values is correct, most likely any pre-imaging timepoint names, "blank" timepoints, and all regional barcode timepoint names.
* `tracingExclusions` should generally be specified, and its value should be a list of timepoints of imaging rounds to exclude from tracing, typically the blank / pre-imaging rounds, and regional barcode rounds. The list can't contain any timepoints not seen in the values of the `imagingRounds`.


## _Parameters_ configuration file
Looptrace uses a configuration file to define values for processing parameters and pointers to places on disk from which to read and write data. 
The path to the configuration file is a required parameter to [run the pipeline](#general-workflow), and it should be created (or copied and edited) before running anything.

### Requirements and suggestions
* Check that each channel setting (often with a `_ch` or `_channel` suffix) matches what's been used in the imaging experiment which generated the data to be processed.
* `xy_nm` and `z_nm` should be adjusted to match your microscope settings (number of nanomerters per step in xy or in z).
* `nuc_method` should be set to "nuclei".
* `nuc_3d` should be absent or set to `False`.
* `analysis_path` should be an _absolute_ path but may use environment and/or user variables.
* `analysis_path` should specify the path to a folder that exists before the pipeline is run.
* `zarr_conversions` should be a mapping from subfolder in the images folder (`--images-folder` when [running from the command-line](#general-workflow)) to new subfolder (1-to-1): the keys are names of subfolders with raw image files (e.g., `.nd2`), and each value will be the new folder with that image data, just reformatted as `.zarr`. 
Typically there will be one entry for the sequential FISH images' folder and another for the nuclei images' folder.
* Check that `spot_wavelength` and `objective_na` have been adjusted to match the microscope and fluorophores used.
* `decon_psf` should be set to `gen`.
* `require_gpu` should be set to `True`.
* `decon_input_name` should likely be the single value in the `zarr_conversions` mapping.
* `reg_input_template` and `reg_input_moving` should likely match each other and should correspond to adding a `_decon` suffix to the value of `decon_input_name`.
* `reg_ref_frame` should be set to something approximately equal to the middle of the imaging timecourse (i.e, midway between first and last timepoint).
* `num_bead_rois_for_drift_correction` should be set to 100 or 200 (number of beads to use for drift correction).
Having bead count fewer than the value will impede processing, but higher values _may_ given a bit better drift correction.
Judge in accordance with how many beads you anticipate having per image.
* `num_bead_rois_for_drift_correction_accuracy` should be set to 100.
* `coarse_drift_downsampling` should be set to 2; use 1 for no downsampling.
* `detection_method` should be set to `dog`, and `spot_threshold` to 15. If using `intensity`, a much higher `spot_threshold` will be needed.
* `subtract_crosstalk` should be set to `False`, rendering `crosstalk_ch` irrelevant.
* `parallelise_spot_detection` should be set to `False`.
* `spot_downsample` should be a small integer, often just 2 or even 1 (no downsampling).
* `spot_in_nuc` should be set to `True`, generally.
* `padding_method` should be set to `edge`.
* `tracing_cores` should be a value no more than the number of CPUs on the machine on which the processing will run.
* `mask_fits` should be set to `False`.
* `roi_image_size` should be a 3-element list and should most likely be (8, 16, 16) or (16, 32, 32).
* `subtract_background` should be set to 0.
* Check that the tracing QC parameters are as desired:
    * For `A_to_BG`, 2 is often a good setting.
    * For `sigma_xy_max`, 150 is often a good setting.
    * For `sigma_z_max`, 400 is often a good setting.
    * For `max_dist`, 800 is often a good setting.
* If you want the Numpy arrays representing the spot images for tracing (the `*.npy` files) to be kept even after zipping, set `keep_spot_images_folder` to `True`.
* Check that the list of `spot_frame` values corresponds to the timepoints (0-based) use for regional barcode imaging in the [imaging rounds config file](#imaging-rounds-configuration-file).


## General workflow
Once you have the [minimal requirements](#minimal-requirements), this will be the typical workflow for running the pipeline:
1. __Login__ to the machine: something like `ssh username@ask-Vince-or-Chris-for-the-machine-domain`
1. __Path creation__: Assure that the necessary filepaths exist; particularly easy to forget are the path to the folder in which analysis output will be placed (the value of `analysis_path` in the parameters config file), and the path to the folder in which the pipeline will place its own files (`-O / --output-folder` argument at the command-line). See the [data layout section](#data-layout-and-organisation).
1. `tmux`: attach to an existing `tmux` session, or start a new one. See the [tmux section](#tmux) for more info.
1. __Docker__: Start the relevant Docker container:
    ```shell
    nvidia-docker run --rm -it -u root -v /groups/gerlich/experiments/.../00XXXX:/home/experiment looptrace:2023-12-12 bash
    ```
1. __Run pipeline__: Once in the Docker container, run the pipeline, replacing the file and folder names as needed / desired:
    ```shell
    python /looptrace/bin/cli/run_processing_pipeline.py --rounds-config /home/experiment/looptrace_00XXXX.rounds.json --params-config /home/experiment/looptrace_00XXXX.params.yaml --images-folder /home/experiment/images_all -O /home/experiment/pypiper_output
    ```
1. __Detach__: `Ctrl+b d` -- for more, see the [tmux section](#tmux).


## `tmux`
In this context, for running the pipeline, think of the _terminal multiplexer_ (`tmux`) as a way to start a long-running process and be assured that an interruption in Internet connectivity (e.g., computer sleep or network failure) won't also be an interruption in the long-running process. If your connection to the remote machine is interrupted, but you've started your long-running process in `tmux`, that process won't be interrupted.

### Basic usage
1. Start a session: `tmux`
1. Detach from the active session: `Ctrl+b d` (i.e., press `Ctrl` and `b` at the same time, and then `d` afterward.)
1. List sessions: `tmux list-sessions`
1. Attach to a session: `tmux attach -t <session-number>`
1. Detach again: `Ctrl+b d`
1. (Eventually) stop a session: `tmux kill-session -t <session-number>`

For more, search for "tmux key bindings" or similar, or refer to [this helpful Gist](https://gist.github.com/mloskot/4285396)/
