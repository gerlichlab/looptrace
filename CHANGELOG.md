# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project will adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
* Moved to using logging to replace `println` in JVM part of the project. See [issue 208](https://github.com/gerlichlab/looptrace/issues/208).

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

