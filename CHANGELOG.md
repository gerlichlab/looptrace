# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project will adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.4.0] - 2024-05-06

### Added
* Added hook (`"checkLocusTimepointCoveringNel"`) for specifying whether or not (`true` / `false`) to check that union of values in `locusGrouping` section of imaging rounds configuration file covers the set of locus spot imaging timepoints in the `imagingRounds` declaration of the same file. The default is `true` (to check for covering). See [Issue 295](https://github.com/gerlichlab/looptrace/issues/295).
* Added pipeline step to set up visualisation (in Napari) of nuclei labels and masks, atop nuclei imaging data. See [Issue 313](https://github.com/gerlichlab/looptrace/issues/313).
* Added pipeline step to set up visualisation (in Napari) of nuclei labels and masks, atop nuclei imaging data. See [Issue 313](https://github.com/gerlichlab/looptrace/issues/311).
* Added pipeline precheck to validate the imaging rounds config. See [Issue 294](https://github.com/gerlichlab/looptrace/issues/294).

### Changed
* Modified and added details on Napari plugin use.
* Updated documentation on how to specify regional spot proximity filtration strategy. See [Issue 310](https://github.com/gerlichlab/looptrace/issues/310).
* Moved to using logging to replace `println` in JVM part of the project. See [Issue 208](https://github.com/gerlichlab/looptrace/issues/208).
* Use just one byte, not two, to store nuclear masks when possible, to save I/O time and space. See [Issue 312](https://github.com/gerlichlab/looptrace/issues/312).
* Updated name and content of pipeline control flow document to better describe "rerun with new parameters" vs. "restart after pipeline halt/fail". See [Issue 293](https://github.com/gerlichlab/looptrace/issues/293).
* Write out nuclear mask visualisation data as soon as it's available, to permit inspection of detection/segmentation right away. See [Issue 302](https://github.com/gerlichlab/looptrace/issues/302).
* Use a `git` dependency on a version of `pypiper` which will support `nofail` behavior for a pipeline phase/stage, linked to [Issue 313](https://github.com/gerlichlab/looptrace/issues/313).
* Allow a locus imaging timepoint to be absent from `locusGrouping` section's values sets' union if that timepoint is also in the `tracingExclusions`. See [Issue 304](https://github.com/gerlichlab/looptrace/issues/304).
* Remove `prefer='threads'` hint to `joblib` for choosing parallelisation backend during coarse drift correction. This aims to minimise the frequency with which this pipeline step will be killed due to insufficient resources.

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
