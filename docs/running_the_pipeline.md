# How to run the main `looptrace` processing pipeline
This document describes the essentials needed to run the main `looptrace` processing pipeline. This is primarily for end-users and does not describe much about the software design or packaging.


## Minimal requirements
To be able to run this pipeline on the lab machine, these are the basic requirements:
* __Have an account on the machine__: you should be able to authenticate with your normal username/password combination used for most authentication within the institute.
* __Be in the `docker` group__: If you've not run something with `docker` on this machine previously, you're most likely not in this group. Ask Vince or Chris to add you.


## Data layout and organisation
* __Main experiment folder__ (`CURR_EXP_HOME` environment variable): On the cluster and on the lab machine, this is often something like `/path/to/experiments/folder/Experiments_00XXXX/00XXXX`, but it could be anything so long as the substructure matches what's expected / defined in the config file.
* __Images subfolder__ (created on lab machine or cluster): something like `images_all`, but just needs to match the value you'll give with the `-I / --images-folder` argument when running the pipeline.
    * _Raw nuclei images_ subfolder: something like `nuc_images_raw`, though just must match the corresponding key in the config file
    * _FISH images_ subfolder: something like `seq_images_raw`, though just must match the corresponding key in the config file
* __Pypiper subfolder__ (created on lab machine or on cluster): something like `pypiper_output`, where pipeline logs and checkpoints are written; this will be passed by you to the pipeline runner through the `-O / --output-folder` argument when you run the pipeline.
* __Analysis subfolder__ (created on lab machine or on cluster): something like `2023-08-10_Analysis01`, though can be anything and just must match the name of the subfolder you supply in the config file, as the leaf of the path in the `analysis_path` value


## Configuration file
Looptrace uses a configuration file to define values for processing parameters and pointers to places on disk from which to read and write data. 
The path to the configuration file is a required parameter to [run the pipeline](#general-workflow), and it should be created (or copied and edited) before running anything.

### Requirements and suggestions
* `analysis_path` should be an _absolute_ path but may use environment and/or user variables.
* `analysis_path` should specify the path to a folder that exists before the pipeline is run.
* `require_gpu` should be set to `True`.
* `decon_psf` should be set to `gen`.
* `xy_nm` and `z_nm` should be adjusted to match your microscope settings (number of nanomerters per step in xy or in z).
* `reg_input_template` and `reg_input_moving` should likely match and should correspond to adding a `_decon` suffix to the value of `decon_input_name`.
* `reg_ref_frame` should be set to something approximately equal to the middle of the imaging timecourse (i.e, midway between first and last timepoint).
* `bead_points` should be set to 100 or 200 (number of beads to use for drift correction).
Having bead count fewer than the value will impede processing, but higher values _may_ given a bit better drift correction.
Judge in accordance with how many beads you anticipate having per image.
* `num_bead_rois_for_drift_correction_accuracy` should be set to 100.
* `coarse_drift_downsampling` should be set to 2; use 1 for no downsampling.
* `detection_method` should be set to `dog`, and `spot_threshold` to 15. If using `intensity`, a much higher `spot_threshold` will be needed.
* `subtract_crosstalk` should be set to `False`, rendering `crosstalk_ch` irrelevant.
* `parallelise_spot_detection` should be set to `False`.
* `spot_downsample` should be a small integer, often just 2 or even 1 (no downsampling).
* `spot_in_nuc` should be set to `False`.
* `padding_method` should be set to `edge`.
* `tracing_cores` should be a value no more than the number of CPUs on the machine on which the processing will run.
* `mask_fits` should be set to `False`.
* `roi_image_size` should be a 3-element list and should most likely be (8, 16, 16) or (16, 32, 32).
* `subtract_background` should be set to 0.
* Check that the tracing QC parameters are as desired
    * For `A_to_BG`, 2 is often a good setting.
    * For `sigma_xy_max`, 150 is often a good setting.
    * For `sigma_z_max`, 400 is often a good setting.
    * For `max_dist`, 800 is often a good setting.
* Check that each channel setting (often with a `_ch` or `_channel` suffix) matches what's been used in the imaging experiment which generated the data to be processed.
* Check that `spot_wavelength` and `objective_na` have been adjusted to match the microscope and fluorophores used.
* Check that the list of `spot_frame` values corresponds to the timepoints (0-based) use for regional barcode imaging.
* Check that the list of `frame_name` values corresponds to how you'd like the probes/frames/timepoints to be labeled.
* Check that the list of `illegal_frames_for_trace_support` values is correct, most likely any pre-imaging timepoint names, "blank" timepoints, and all regional barcode timepoint names.
* __If multiplexing__: Check that the `regional_spots_grouping` specifies the correct groupings of regional barcode imaging timepoints to reflect the regional barcodes which _ARE_ allowed to be in close proximity (in violation of `min_spot_dist`).


## General workflow
Once you have the [minimal requirements](#minimal-requirements), this will be the typical workflow for running the pipeline:
1. __Login__ to the machine: something like `ssh username@ask-Vince-or-Chris-for-the-machine-domain`
1. __Path creation__: Assure that the necessary filepaths exist; particularly easy to forget are the path to the folder in which analysis output will be placed (the value of `analysis_path` in the config file), and the path to the folder in which the pipeline will place its own files (`-O / --output-folder` argument at the command-line). See the [data layout section](#data-layout-and-organisation).
1. `tmux`: attach to an existing `tmux` session, or start a new one. See the [tmux section](#tmux) for more info.
1. __Docker__: Start the relevant Docker container:
    ```shell
    nvidia-docker run --rm -it -u root -v /groups/gerlich/experiments/.../00XXXX:/home/experiment looptrace:2023-12-12 bash
    ```
1. __Run pipeline__: Once in the Docker container, run the pipeline, replacing the file and folder names as needed / desired:
    ```shell
    python /looptrace/bin/cli/run_processing_pipeline.py -C /home/experiment/looptrace_00XXXX.yaml -I /home/experiment/images_all -O /home/experiment/pypiper_output
    ```
1. __Detach__: `Ctrl+b d` -- for more, see the [tmux section](#tmux).


## tmux
In this context, for running the pipeline, think of the _terminal multiplexer_ (`tmux`) as a way to start a long-running process and be assured that an interruption in Internet connectivity (e.g., computer sleep or network failure) won't also be an interruption in the long-running process. If your connection to the remote machine is interrupted, but you've started your long-running process in `tmux`, that process won't be interrupted.

### Basic usage
1. Start a session: `tmux`
1. Detach from the active session: `Ctrl+b d` (i.e., press `Ctrl` and `b` at the same time, and then `d` afterward.)
1. List sessions: `tmux list-sessions`
1. Attach to a session: `tmux attach -t <session-number>`
1. Detach again: `Ctrl+b d`
1. (Eventually) stop a session: `tmux kill-session -t <session-number>`

For more, search for "tmux key bindings" or similar, or refer to [this helpful Gist](https://gist.github.com/mloskot/4285396)/
