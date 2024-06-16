# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project will adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
