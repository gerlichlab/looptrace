# Visualising spots
Within the course of a pipeline run, `looptrace` "detects" FISH spots and computes various properties of them. 
Some of this information is available for visualisation.
Visualisation of spots and their properties, overlaying the image data, can be a good examination of how well something like spot detection or QC filtering is doing.

## Locus-specific spots
`looptrace` prepares subimages and "points" corresponding to spot centers which are ready to be jointly visualised in the [`napari` image viewer](https://napari.org/stable/).
This tool is typically what we use for visualisation of spots and so we've tested more there, but you may be able to view the spot images overlaid with points layers in a similar image viewing program.

### General process
1. Ensure `napari` is available as a command in your terminal by typing `which napari` and seeing that you get a nonempty result displayed, pointing to an installation of the `napari` executable.
1. If `napari` is not yet available, you have a couple of options. You may follow the [installation instructions](https://napari.org/stable/tutorials/fundamentals/quick_start.html#installation) from the `napari` project itself.
Alternatively, if you have [Nix package manager](https://nixos.org/download) installed on your machine, you may use this project's Nix shell. If using the Nix shell, type `nix-shell --arg interactive true` from the root folder of this project. Note that we've only tested this on M1/M2 Macs.
1. You will also need some [PyQt](https://riverbankcomputing.com/software/pyqt/intro)-related things available behind-the-scences, but if you've followed the Napari installation guide or are using this project's Nix shell on a newer Mac, you should be all set.
1. Open the Napari viewer by typing `napari` into a terminal where the command is available.
1. In `Finder` or a similar file browser program, open the analysis subfolder of an experiment folder to which `looptrace` has written results, and navigate to the `locus_spot_images` subfolder, and then click-and-drag the image data for the field of view you're interested in and the correspinding QC pass/fail files into the Napari window.
1. For each points layers in turn, select all the point with `Shift+A`, and then set color, shape, size, etc. as desired.
1. When you want to view a different field of view (FOV), simply delete the layers within the Napari viewer, and the repeat the process with the image file and points files for the new FOV.

