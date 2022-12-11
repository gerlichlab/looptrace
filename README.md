# Loop tracing in python.

LoopTrace is a Python package for performing procedures of chromatin tracing in Python.

## Installation

The simplest way to install the package and all dependencies is to clone the repository and create a new environment using the 
environment.yml file. There are several optional dependencies, these can be installed separately (see below), or directly during installation by using environment_full.yml as the environment file instead. 
In a terminal (e.g. a miniconda/Anaconda prompt):

```bash
git clone https://git.embl.de/grp-ellenberg/looptrace
cd looptrace
conda env create -f environment.yml
python setup.py install
conda activate looptrace
```
There are several optional packages that can be added depending on use case:

Deconvolution of spots during tracing:
```bash
pip install flowdec[tf_gpu]
```

Detection of nuclei:
```bash
pip install cellpose
```

Comparison of FISH procedures:
```bash
pip install imreg_dft
```

For analysis, visualization and plotting of trace data:
```bash
conda install plotly nbformat ipykernel nb_conda
pip install seaborn
```

For controlling a microfluidics system, pyserial is required:
```bash
pip install pyserial
```

In addition, Ilastik can be used

## Basic usage (tracing):
First edit the config YAML file to provide input and output directories and other parameters. A documented example file is found in examples/trace_analysis_config.yaml.

Once edited, start the GUI from the terminal:

```bash
python bin\looptrace_gui.py
```
- In the GUI, browse to the experiment config file configured above, and press "Initialize". 
- If desired, raw imaging files can be converted to a high-performance zarr archive with "Save to zarr". If used, remember to update config file upon the next run. Once saved the zarr files are automatically used later in the workflow.
- If config parameters are changed, press "Reload config" to update these in the GUI.
- Images from the experiment can be viewed with "View images for tracing".
- "Run drift correction" will calculate a drift correction for all positions and hybridizations based on settings in the config file. Output is save in the output directory as "prefix_drift_correction.csv". 
- The standard drift correction file should be pre-selected after calculating the drift, otherwise an alternative one can be used.
- The drift corrected images can now be viewed with "View DC images". Only the course (pixel-level) drift correction is applied when viewing.
- If nuclei have been imaged, these can be segmented and used for filtering FISH spots, and classifed using Ilastik. A separate Ilastik installation and pre-trained model need to be specified in the config file if classification is to be used.
- If ROIs for FISH spots are detected in another software (e.g. ImageJ), or in a previous session, these can now be loaded (remember to press "Load existing ROIs" after selecting the file.)
- If new ROIs are to be detected, the threshold for this is set in the config file. Preview the threshold selection with "Preview ROIs", and adjust as necessary in the config file and press "Reload config".
- Once the threshold is set, press "Detect new ROIs" to run the detection on all frames. These are saved in the output folder er "prefix_rois.csv".
- Once loaded or detected, ROIs can be viewed with "View ROIs", either on non-corrected or drift-correction images. In napari, the point tool can be used to add or remove detected ROIs manually as desired. The ROI file is updated accordingly.
- "Refilter ROIs" reassignes nucleus identities and classes (if these were generated earlier) after manual QC of ROIs.
- Once drift correction and ROIs have been performed (or loaded from a previous session), "Run tracing" will perform the fitting of sequential FISH spots as specified in the config file. The output coordinates and single spot images are saved in the output folder ("prefix_traces.csv" and "prefix_imgs.tif").

Once the tracing is done, the data can be further analyzed for example using iPython notebooks. See examples of the full analysis as well as several full datasets at https://www.ebi.ac.uk/biostudies/studies/S-BIAD59

## Authors
Written and maintained by Kai Sandvold Beckwith (kai.beckwith@embl.de), Ellenberg group, CBB, EMBL Heidelberg.
See https://www-ellenberg.embl.de/. 

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Citation
Please cite our paper.
