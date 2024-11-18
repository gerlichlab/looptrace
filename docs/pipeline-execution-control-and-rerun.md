<!--- DO NOT EDIT THIS GENERATED DOCUMENT DIRECTLY; instead, edit generate_execution_control_document.py --->
# Controlling pipeline execution

## Overview
The main `looptrace` processing pipeline is built with [pypiper](https://pypi.org/project/piper/).
One feature of such a pipeline is that it may be started and stopped at arbitrary points.
To do so, the start and end points must be specified by name of processing stage.

To __start__ the pipeline from a specific point, use `--start-point <stage name>`. Example:
```
python run_processing_pipeline.py \
    --rounds-config rounds.json \
    --params-config params.yaml \
    --images-folder images_folder \
    --pypiper-folder pypiper_output \
    --start-point spot_detection \
    --stop-before tracing_QC
```

To __stop__ the pipeline...<br>
* ...just _before_ a specific point, use `--stop-before <stage name>`.
* ...just _after_ a specific point, use `--stop-after <stage name>`.

## Rerunning the pipeline...
When experimenting with different parameter settings for one or more stages, it's common to want to rerun the pipeline from a specific point.
Before rerunning the pipeline with the appropriate `--start-point` value, take care of the following:

1. __Analysis folder__: It's wise to create a new analysis / output folder for a rerun, particularly if it corresponds to updated parameter settings.
1. __Parameters configuration file__: It's wise to create a new parameters config file for a rerun if the rerun includes updated parameter settings. 
Regardless of whether that's done, ensure that the `analysis_path` value corresponds to the output folder you'd like to use.
1. __Imaging rounds configuration file__: If a new analysis for the same experimental data affects something about the imaging rounds configuration, 
e.g. the minimum separation distance required between regional spots, you may want to create this config file anew, copying the old one and updating 
the relevant parameter(s).
1. __Pipeline (pypiper) folder__: You should create a new pypiper folder _for a rerun with new parameters_. 
This is critical since the semaphore / checkpoint files will influence pipeline control flow.
You should copy to this folder any checkpoint files of any stages upstream of the one from which you want the rerun to begin.
Even though `--start-point` should allow the rerun to begin from where's desired, if that's forgotten the checkpoint files should save you.
For a _restart_--with the same parameters--of a stopped/halted/failed pipeline run, though, you should generally reuse the same pypiper folder as before. 

Generate an empty checkpoint file for each you'd like to skip. 
Simply create (`touch`) each such file `looptrace_<stage>.checkpoint` in the desired pypiper output folder.
Below are the sequential pipeline stage names.

### Pipeline stage names
* imaging_rounds_validation
* pipeline_precheck
* zarr_production
* psf_extraction
* deconvolution
* nuclei_detection
* nuclei_drift_correction
* nuclear_masks_visualisation_data_prep
* move_nuclear_masks_visualisation_data
* drift_correction__coarse
* bead_roi_generation
* bead_roi_detection_analysis
* bead_roi_partition
* drift_correction__fine
* drift_correction_accuracy_analysis
* drift_correction_accuracy_visualisation
* spot_detection
* spot_merge_determination
* spot_merge_execution
* spot_proximity_filtration
* spot_nucleus_filtration
* trace_id_assignment
* regional_spots_visualisation_data_prep
* spot_counts_visualisation__regional
* spot_bounding
* spot_extraction
* spot_zipping
* spot_background_zipping
* tracing
* trace_annotation
* spot_region_distances
* tracing_QC
* spot_counts_visualisation__locus_specific
* pairwise_distances__locus_specific
* pairwise_distances__regional
* locus_specific_spots_visualisation_data_prep
* locus_spot_viewing_prep