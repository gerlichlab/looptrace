# Visualising spots
Within the course of a pipeline run, `looptrace` "detects" FISH spots and computes various properties of them. 
Some of this information is available for visualisation.
Visualisation of spots and their properties, overlaying the image data, can be a good examination of how well something like spot detection or QC filtering is doing.

The default Nix shell for this project provides Napari and a couple of Napari plugins for viewing outputs that this project's pipeline produces. 
Refer to those specific projects for more about [viewing nuclei and nuclear masks](https://github.com/gerlichlab/nuclei-vis-napari) and about [visualising locus spots](https://github.com/gerlichlab/looptrace-loci-vis).

