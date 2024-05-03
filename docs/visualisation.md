# Visualising nuclei and spots
Within the course of a pipeline run, `looptrace` "detects" FISH spots and computes various properties of them. 
Some of this information is available for visualisation.
Visualisation of spots and their properties, overlaying the image data, can be a good examination of how well something like spot detection or QC filtering is doing.

Similarly, nuclei are detected and the corresponding image regions are used to filter (positively select) spots to be used for analysis. 
The performance of the nuclei detection step (and concomitantly, the quality of the filtration of FISH spots) can be assessed qualitatively by inspecting nucleus imaging data and the detected regions/masks.

The default Nix shell for this project provides Napari and a few Napari plugins for viewing outputs that this project's pipeline produces. 
Refer to those specific projects for more about [viewing nuclei and nuclear masks](https://github.com/gerlichlab/nuclei-vis-napari), [visualising regional spots](https://github.com/gerlichlab/looptrace-regionals-vis), and [visualising locus spots](https://github.com/gerlichlab/looptrace-loci-vis).

