<!--- DO NOT EDIT THIS GENERATED DOCUMENT DIRECTLY; instead, edit generate_excution_control_document.py --->
# Controlling pipeline execution
## Overview
The main `looptrace` processing pipeline is built with [pypiper](https://pypi.org/project/piper/).
One feature of such a pipeline is that it may be started and stopped at arbitrary points.
To do so, the start and end points must be specified by name of processing stage.

To __start__ the pipeline from a specific point, use `--start-point <stage name>`. Example:
```python
python run_processing_pipeline.py -C conf.yaml -I images_folder -O pypiper_output --start-point spot_detection --stop-before tracing_QC
```

To __stop__ the pipeline...<br>
* ...just _before_ a specific point, use `--stop-before <stage name>`.
* ...just _after_ a specific point, use `--stop-after <stage name>`.

## Restarting the pipeline...
When experimenting with different parameter settings for one or more stages, it's common to want to restart the pipeline from a specific point.
Before rerunning the pipeline with the appropriate `--start-point` value, take care of the following:

1. __Analysis folder__: It's wise to create a new analysis / output folder for a restart, particularly if it corresponds to updated parameter settings.
1. __Configuration file__: It's wise to create a new config file for a resart if it corresponds to updated parameter settings. 
Regardless of whether that's done, ensure that the `analysis_path` value corresponds to the output folder you'd like to use.
1. __Pipeline (pypiper) folder__: You should create a new pypiper folder for a restart with new parameters.
This is critical since the semaphore / checkpoint files will influence pipeline control flow.
You should copy to this folder any checkpoint files of any stages upstream of the one from which you want the restart to begin.
Even though `--start-point` should allow the restart to begin from where's desired, if that's forgotten the checkpoint files should save you.

Generate an empty checkpoint file for each you'd like to skip. 
Simply create (`touch`) each such file `looptrace_<stage>.checkpoint` in the desired pypiper output folder.
Below are the sequential pipeline stage names.

### Pipeline stage names
* pipeline_precheck
* zarr_production
* nuclei_detection
* psf_extraction
* deconvolution
* drift_correction__coarse
* bead_roi_generation
* bead_roi_detection_analysis
* bead_roi_partition
* drift_correction__fine
* drift_correction_accuracy_analysis
* drift_correction_accuracy_visualisation
* spot_detection
* spot_proximity_filtration
* spot_counts_visualisation__regional
* spot_nucleus_filtration
* spot_bounding
* spot_extraction
* spot_zipping
* tracing
* spot_region_distances
* tracing_QC
* spot_counts_visualisation__locus_specific
* pairwise_distances__locus_specific
* pairwise_distances__regional