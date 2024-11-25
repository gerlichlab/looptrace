# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project will adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.11.0] - 2024-11-25

### Added
* "Merger" of ROIs from different timepoints, such that structures which should be analyzed together spatially but whose components are targeted and imaged at different timepoints are properly grouped. 
See [Issue 279](https://github.com/gerlichlab/looptrace/issues/279).
* Cross-channel signal analysis extraction. See [Issue 337](https://github.com/gerlichlab/looptrace/issues/337).

### Fixed
* Ensure that same-timepoint-ROI merge is done in connected-components fashion, rather than simply by pairwise-distance-defined proximity. 
See [Issue 368](https://github.com/gerlichlab/looptrace/issues/368).
* Make ROI IDs downstream of regional spot filtration correctly map back to the original spots. 
See [Issue 372](https://github.com/gerlichlab/looptrace/issues/372).

### Changed
* Scala is now version 3.5.2.
* `sbt` is now version 1.10.5.
* `uJson` and `uPickle` are now at version 4.0.2.
* License is now Apache 2.0, due to our use of `iron` and `gerlib`.
* Moved the `AtLeast2[C[*], E]` data type to `gerlib`, to be released with version 0.3 of that library.
* Changed `region1` and `region2` to (isomorphic) `timepoint1` and `timepoint2`, respectively, in output file for regional spot pairwise distances
* Changed `inputIndex1` and `inputIndex2` to `roiId1` and `roiId2`, respectively, in output file for regional spot pairwise distances. 
These now point back to specific regional spots directly (by ID number), rather than simple line number/index. See [Issue 362](https://github.com/gerlichlab/looptrace/issues/362).
* Now depending on updated plugin versions for visualisation:
    * v0.1.3 of the nuclei visualisation plugin
    * v0.3.1 of the regional spots visualisation plugin
    * v0.2.1 of the locus spots visualisation plugin

## [v0.10.1] - 2024-11-11

### Fixed
* Fixed the incorrectness of the merged regional spot centers and bounding boxes in the `*_rois.proximity_accepted.nuclei_labeled.csv` and `*_rois.proximity_accepted.nuclei_filtered.csv`. See [Issue 353](https://github.com/gerlichlab/looptrace/issues/353), [Issue 355](https://github.com/gerlichlab/looptrace/issues/355), and [Issue 356](https://github.com/gerlichlab/looptrace/issues/356).
* Correctly applying drift correction to regional spot coordinates. See [Issue 329](https://github.com/gerlichlab/looptrace/issues/329).
* Unit of measure for the regional spot pairwise distances is now nanometers, rather than pixels (variably defined) as previously. See [Issue 359](https://github.com/gerlichlab/looptrace/issues/359).

### Changed
* Using `fs2` to write out regional pairwise distance results to CSV
* Using `fs2` to write out locus pairwise distance results to CSV

### Known Issues
* ROI IDs starting from the spot table extraction step will most likely not correctly map back to the detected spot, probably predating the 0.10.x line. See [Issue 371](https://github.com/gerlichlab/looptrace/issues/371).

## [v0.10.0] - 2024-10-22

### Added
* "Merge" ROIs which come from spots detected from the same regional barcode imaging timepoint, by defining a new ROI around the midpoint of the line segment between their centroids. See [Issue 279](https://github.com/gerlichlab/looptrace/issues/285)

### Changed
* Swapped the order of the proximity-based filtration and the through-nuclei filtration; the filtration through nuclei now happens first.
* Bumped up to Scala 3.5.1.

### Known Issues
* ROI IDs starting from the spot table extraction step will most likely not correctly map back to the detected spot, probably predating the 0.10.x line. See [Issue 371](https://github.com/gerlichlab/looptrace/issues/371).

## [v0.9.1] - 2024-07-29

### Fixed
* Make spot detection not crash due to use of method as property in spot detection.

## [v0.9.0] - 2024-07-19

### Changed
* _Minimum Python version is now 3.11_!
* The `_nuclear_masks_visualisation/*.nuclear_masks.csv` files are now renamed as `_nuclear_masks_visualisation/*.nuclei_centroids.csv`, to better reflect the data content of these files and how they're used for the visualisation of the nuclear masks.
* _Removed from API_: `looptrace.integer_naming.get_channel_name_short` and `looptrace.integer_naming.get_time_name_short`
* _Removed_ the program to "squeeze" FOVs' numbering into a contiguous subsequence at the beginning of the natural numbers. 
The thinking is that the original naming/numbering of the FOVs should be preserved.
* Sorting ND2 and ZARR stacks by name of position / field of view.

### Fixed
* Preserve field-of-view-based naming of nuclear mask centroid files, even when a field of view has been removed. See [Issue 285](https://github.com/gerlichlab/looptrace/issues/285) and [Issue 344](https://github.com/gerlichlab/looptrace/issues/344).
* Catch correct error types at a couple places where a more generic / builtin exception was being caught, rather than the narrower domain-specific one which would've been thrown.

### Known Issues
* Spot detection will likely crash, due to a method being used like a property.

## [v0.8.0] - 2024-07-12

### Fixed
* Nuclear mask ZARR arrays will now have names matched to their original field of view name, not simply contiguous initial subsequence of natural numbers. See [Issue 344](https://github.com/gerlichlab/looptrace/issues/344).

### Changed
* The signature of `image_io.images_to_ome_zarr` changes to permit only a collection of per-field-of-view arrays, and the function no longer accepts a data type argument but rather infers the data type from the given data, raising an error if the same type isn't inferred for every array in the given collection.

### Known Issues
* The field of view numbering will may still be off in the `_nuclear_masks_visualisation/*.nuclear_masks.csv` files, if a field of view was removed from the processing, e.g. on account of bad data. 
This _may_ not cause any problems _per se_ for the visualisation, but know that it _could_, and that the numbers won't correspond to those of the initial nuclei imaging files w.r.t. position / field of view.

## [v0.7.0] - 2024-07-11

### Changed
* Regional imaging timepoints from the `ImageHandler` will now be sorted numerically, rather than by the order in which they're declared in the imaging rounds configuration file. 
Essentially, this means that it will be allowed to declare rounds out of order (intentionally or by mistake), but the correct time sequence of rounds should still result.
* Make it illegal to provide the (regional) spot detection imaging timepoints in the parameters config file. 
They should be defined in and read from only the imaging rounds configuration file. See [Issue 338](https://github.com/gerlichlab/looptrace/issues/338).
* Removed the spot detection gridsearch CLI program

## [v0.6.0] - 2024-07-10

### Changed
* Core data types now taken from `gerlib`
* Reorganisation of `given`s / implicit instances of key typeclasses, to mirror the `cats` structure

## [v0.5.3] - 2024-06-20

### Added
* `coarse_dc_backend` and `coarse_dc_cpu_count` are now avaible as keys in the parameters configuration file, to control which backend that `joblib`'s parallel processing will use, and how many CPUs will be used, respectively.

### Changed
* By default, coarse drift correction will now use the `'threading'` backend for `joblib` and only half of the available CPUs, attempting to alleviate resource pressure. 
See [Issue 332](https://github.com/gerlichlab/looptrace/issues/330).

## [v0.5.2] - 2024-06-16
This is strictly a bugfix release.

### Fixed
* `str.lstrip` is replaced by `str.removeprefix` where appropriate. See [Issue 330](https://github.com/gerlichlab/looptrace/issues/330).

## [v0.5.1] - 2024-05-31

### Changed
* The processing pipeline now will permit overwrite of existing outputs of the tracing QC, rather than crashing.
* Bumped `os-lib` dependency to 0.10.2.

## [v0.5.0] - 2024-05-31

### Added
* Added a tool--a simple [command-line program](./bin/cli/analyse_bead_discard_reasons.py)-- to analyse the reason(s) that each fiducial bead ROI is discarded (_if_ it's discarded).
* Added a tool-- a simple [command-line program](./bin/cli/squeeze_list_of_field_of_view_zarr.py)--to make a folder of per-FOV ZARRs into a contiguous sequence at the start of the natural numbers, after having removed certain problematic FOVs' ZARRs.
* Added pipeline step, corresponding to a [command-line program](./bin/cli/locus_spot_visualisation_data_preparation.py) to set up visualisation (in Napari) of locus specific spots, atop pixel volume data.
* Added `spot_in_nuc` property to image handler objects.
* Added some explanations of the different static test cases, to document the underlying logic / contracts being tested as upheld or not.
* Added some documentation about the `locusGrouping` section of the imaging rounds configuration file.
* Added direct (in-memory, not through JSON) PBT for enforcement of required exclusion of regional timepoint from its locus times collection.
* Storage of background subtracted from true pixel values for each ROI, see [Issue 322](https://github.com/gerlichlab/looptrace/issues/322).

### Changed
* Depend on new version of Pypiper (0.14.2), which allows any pipeline stage to be marked as `nofail` (by passing `nofail=True` to its constructor), so that if something goes wrong while running that stage, it won't fail the whole pipeline (i.e., subsequent processing is allowed to continue).
* Use the `locusGrouping` section of the imaging rounds config to only extract image volumes for tracing from timepoints in which it "makes sense". Specifically, for a given regional spot, only extract corresponding pixels from imaging timepoints of loci associated with that region. Because of this, the locus spot visualisation data also becomes one-ZARR-per-regional-timepoint, rather than one-ZARR-per-FOV. See [Issue 237](https://github.com/gerlichlab/looptrace/issues/237).
* Changed config hook `"checkLocusTimepointCoveringNel"` to `"checkLocusTimepointCovering"`. See [Issue 295](https://github.com/gerlichlab/looptrace/issues/295) and [Issue 227](https://github.com/gerlichlab/looptrace/issues/237).
* Pool of spots for regional pairwise distance computations now depends on `spot_in_nuc` config value (default `False`, but often `True`).
* Timepoints are redefined/reindexed on a per-regional-timepoint basis, so that the time dimension of data for visualisation with the Napari plugin is appropriately shrunk, to complement how the `locusGrouping` section of the imaging rounds configuration file is now used to more intelligently extract data for tracing.
* Made several members of `looptrace.Tracer` into `property` values rather than direct attributes, so that data aren't computed/stored as much when not needed. See [Issue 303](https://github.com/gerlichlab/looptrace/issues/303).
* Now including the true trace ID in the locus spots visualisation points files (e.g. `P0001.qcpass.csv`, to check if something seems amiss). Previously only the reindexed (after sorting in ascending order) trace ID was included.
* Data for visualisation with the locus spots plugin is now after having subtracted background, as applicable. See [Issue 324](https://github.com/gerlichlab/looptrace/issues/324).
* Proper handling of potentially high pixel values when mapping from unsigned integer to signed integer type during background subtraction for tracing locus spots. See [Issue 325](https://github.com/gerlichlab/looptrace/issues/325).

## [v0.4.1] - 2024-05-24
This is a ___bugfix_ release__.

### Changed
* Fix `zarr` dependency at v2.17.2, to avoid problems during conversion of ND2 to ZARR. See [Issue 321](https://github.com/gerlichlab/looptrace/issues/321).

## [v0.4.0] - 2024-05-06

### Added
* Added hook (`"checkLocusTimepointCoveringNel"`) for specifying whether or not (`true` / `false`) to check that union of values in `locusGrouping` section of imaging rounds configuration file covers the set of locus spot imaging timepoints in the `imagingRounds` declaration of the same file. The default is `true` (to check for covering). See [Issue 295](https://github.com/gerlichlab/looptrace/issues/295).
* Added pipeline step to set up visualisation (in Napari) of nuclei labels and masks, atop nuclei imaging data. See [Issue 313](https://github.com/gerlichlab/looptrace/issues/313).
* Added pipeline precheck to validate the imaging rounds config. See [Issue 294](https://github.com/gerlichlab/looptrace/issues/294).
* Remove `prefer='threads'` hint to `joblib` for choosing parallelisation backend during coarse drift correction. This aims to minimise the frequency with which this pipeline step will be killed due to insufficient resources.

### Changed
* Modified and added details on Napari plugin use.
* Updated documentation on how to specify regional spot proximity filtration strategy. See [Issue 310](https://github.com/gerlichlab/looptrace/issues/310).
* Moved to using logging to replace `println` in JVM part of the project. See [Issue 208](https://github.com/gerlichlab/looptrace/issues/208).
* Use just one byte, not two, to store nuclear masks when possible, to save I/O time and space. See [Issue 312](https://github.com/gerlichlab/looptrace/issues/312).
* Updated name and content of pipeline control flow document to better describe "rerun with new parameters" vs. "restart after pipeline halt/fail". See [Issue 293](https://github.com/gerlichlab/looptrace/issues/293).
* Write out nuclear mask visualisation data as soon as it's available, to permit inspection of detection/segmentation right away. See [Issue 302](https://github.com/gerlichlab/looptrace/issues/302).
* Use a `git` dependency on a version of `pypiper` which will support `nofail` behavior for a pipeline phase/stage, linked to [Issue 313](https://github.com/gerlichlab/looptrace/issues/313).
* Allow a locus imaging timepoint to be absent from `locusGrouping` section's values sets' union if that timepoint is also in the `tracingExclusions`. See [Issue 304](https://github.com/gerlichlab/looptrace/issues/304).

### Known issues
* The heatmaps of counts of usable bead ROIs will be wrong in some cases; specifically, if there are null values in place of empty strings in the table-like file for any (FOV, time) pair, the corresponding bead ROI will be _counted in the visualisation_ as unusable, even though they will be eligible for actual use.
* The conversion of ND2 to ZARR will not work. Please use v0.4.1. Alternatively, you may run just the ZARR conversion on v0.3.1 or v0.4.1, and then return to this version for the rest of processing.

## [v0.3.1] - 2024-04-22
This is a __bugfix release__ for `2024-04-12a`.

### Fixed
* Make `minimumPixelLikeSeparation` required _if and only if_ the filtration strategy semantic is _not_ `UniversalProximityPermission`. See [Issue 308](https://github.com/gerlichlab/looptrace/issues/308).

## [v0.3.0] - 2024-04-21
 
### Changed
* Provide both visualisation plugins for `napari` as default dependencies in the Nix shell.
* `locus_spot_images` subfolder changes to `locus_spots_visualisation`, containing both the ZARR-stored spot images pixel data, and the CSV-stored QC pass/fail points data.
* Bump `gertils` dependency up to v0.4.4.
* Bump `spotfishing` dependency up to v0.1.0.

## [All previous versions]
Previous versions were released with a date tag and sometimes a letter suffix indicating which subversion in a sequence of iterations--based on the same set of fundamental changes--that the release was.
