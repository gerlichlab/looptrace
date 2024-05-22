# `looptrace` imaging rounds configuration file
The imaging rounds configuration file is where, in ___JSON_ format__ you tell `looptrace` about the rounds of imaging you did during a sequential FISH experiment.
It's from this file that a name becomes associated with each imaging round, rounds are conceptually distinguished from one another (e.g., blank vs. regional or local FISH), and the relationships between rounds are defined for analysis.

## Requirements and suggestions

### Proximity filtration section: `proximityFilterStrategy`
This section specifies how to filter regional barcode spots for being too close together. 
When doing filtration, note that the threshold for minimum separation distance must be violated in all three dimensions for spots to be considered proximal, or "neighboring". 
This is because spots that are still well separated in at least one dimension are not really close together in how we normally conceive of distance. 
Note, however, that because of how this threshold is used in the filtration, it's not a "distance" in the typical Euclidean sense in which we usually think.

Note also that there's some unfortunate asymmetry between $z$ vs. $x$ and $y$ that's not analogously asymmetrically treated by the software. 
More specifically, although image acquisition is generally conducted in much more _discrete_ fashion in $z$ than in $x$ and $y$ (generally far fewer planes/slices in $z$ being imaged, compared to the number of pixels which comprise the length and width of each slice), there's a _single_ threshold value, applied identically in every dimension. This means that in general, you'll want to choose a threshold for minimum separation that's relatively small; in particular, it would make little sense to have this value be any more than about half of the "depth" (number of slices) in $z$ that you've acquired.

__Other things to note__:
* _Field of view_ matters: For obvious reason, two spots could have identical coordinates but remain even after filtration, and this would be because they're from _different fields of view_. The coordinate system is specific to each field of view, therefore neighbors/proximity consideration is done only _within each FOV_.
* _Imaging timepoint_ matters: Two spots from the same imaging timepoint, in the same FOV, never cancel each other for proximity; instead, this special case of proximal spots will be handled by attempting to merge the ROIs for the respective spots.

__Guidance__
* `proximityFilterStrategy` is required and should be a mapping, with at _minimum_ a `semantic` key.
* Other key-value pairs are _conditionally_ required or prohibited, depending on the value given for `semantic`.
* The value for the semantic must be one of the following:
    * `"UniversalProximityPermission"`: This says that _no_ proximity-based filtration of regional spots should be done, essentially making the proximity filtration step of the pipeline a _no-op_. This behavior corresponds to what, in previous versions of the software, would have been achieved by setting the minimum spot sepearation distance (`min_spot_dist`) to 0. Because of the meaning of this value, _nothing additional_ is permitted in the section.
    * "UniversalProximityProhibition": This says that _any_ spots which are closer than some threshold should mutually cancel one another. This _requires minimum separation distance_, to be provided as a positive number for the `minimumPixelLikeSeparation` key.
    * `"SelectiveProximityPermission"`: This says that spots from imaging timepoints which are grouped together may violate the minimum separation threshold. Put differently, the _grouping grants permission_; that is, spots from timepoints which are grouped together are given permission to violate the minimum separation threshold. As such, this requires `groups` in addition to `minimumPixelLikeSeparation`. The value for `groups` must be a _list of lists_, with each element being the integer corresponding to a _regional_ barcode imaging round, in the [imaging rounds section](#imaging-rounds). This strategy is useful, for example, if you're imaging at different timepoints DNA regions that you expect to be in very close proximity (e.g., to to being very close in genomic distance).
    * `"SelectiveProximityProhibition"`: This is as for the `"SelectiveProximityPermission"`, but says instead that spots from timepoints which are grouped together are the ones which should mutually exclude each others' spots if they're too close. This strategy is useful, for example, when you're multiplexing and need the regional barcodes to disambiguate loci being targeted with the same locus-specific barcode (i.e., when 2 separate barcodes function jointly as a uniue identifier of a locus).
* If you're using a strategy for which `groups` is specified, the value for `groups` must satisfy these properties:
    * _No repeats_: Each group must have no repeat value.
    * _No overlaps_: The groups must be disjoint.
    * _Covering_: The union of the groups must cover the set of regional round timepoints from the `imagingRounds`.
    * _Covered_: The set of regional round timepoints from the `imagingRounds` must cover the union of groups.

<a href="imaging-rounds"></a>
### Imaging timepoints sequence: `imagingRounds`
This section specifies the time sequence of imaging rounds, with details about each round like whether it was for a regional barcode, if it was a blank, or if it was a repeat.

__Guidance__
* `imagingRounds` must be present, and it enumerates the imaging rounds that comprise the experiment. 
For every entry, `time` is required (as a nonnegative integer), and either `probe` or `name` is required.
No `time` value can be repeated, and the timepoints should form the sequence (0, 1, 2, ..., *N*-1), where *N* is the number of imaging rounds in the experiment.
Each value in this array represents a single imaging round; each round is one of 3 cases:
    * _Blank_ round: specify `time` as a nonnegative integer, and `name` as a string; set `isBlank` to `true`.
    * _Locus_ FISH round: specify `time` as a nonnegative integer, omit `isBlank` or set it to `false`, and specify `probe` as a string. If this is a repeat of some earlier timepoint, specify `repeat` as a positive integer. `name` can be specified or inferred from `probe` and `repeat`; `isRegional` must be absent or `false`.
    * _Regional_ FISH round: this is as for a locus-specific round, but ensure that `isRegional` is set to `true`, and this cannot have a `repeat` (omit it.)
* For any non-regional round, `repeat` may be included with a natural number as the value. In that case, if `name` is absent (derived from `probe`), the suffix `_repeatX` will be added to the `probe` value to determine the name.
* In general, if `name` is absent, it's set to `probe` value (though this is not possible for a blank round, which is why `name` is required when `isBlank` is `true`).
* The set of names for the rounds of the imaging sequence must have no repeated value (including a collision of a derived value with some explicit or derived value).
* `locusGrouping` _can be_ present as a top-level key, if and only if there's one or more locus imaging rounds in the `imagingRounds`.
    * Each key must be a string, but specifying the timepoint of one of the _regional_ rounds from the `imagingRounds` sequence.
    * Each value is a list of locus imaging timepoints associated with the regional timepoint to which it's keyed.
    * The values in the lists must be such that the union of the lists is the set of locus imaging timepoints from the `imagingRounds` sequence.
    * The values lists should have unique values, though as an ensemble they may not necessarily be disjoint.
* `checkLocusTimepointCovering` should generally be absent or set to `true` (the default). 
If, however, your experimental design is such that you _don't_ want to trace all of the locus-specific timepoints, then you should set this to `false` so that a `locusGrouping` section which _doesn't_ cover all of the locus-specific timepoints in the `imagingRounds` section of that file won't be considered an error.

### Relating regional and locus-specific timepoints: `locusGrouping`
This section declares how the regional nad locus-specific timepoints from the [imaging rounds sequence](#imaging-rounds) relate to one another. More specifically, for each regional barcode timepoint, you declare a corresponding collection of locus-specific imaging timepoints. This, determines which image volumes are extracted for which regional spot, and therefore how traces are generated as well as how downstream analysis and visualisation works.

This section _may be absent or empty_, in which case the every timepoint is fit/traced for every regional timepoint; call this "all-by-all", or "universal tracing", maybe, as a way to quickly conceptualise what omission of this section implies.

If _present_, however, the data in this section _must_ comply with certain __rules__:
* The data must be a _JSON object_ (key-value) mapping.
* Each key must be a (quoted) regional barcode imaging timepoint (corresponding to an entry in the [imaging rounds section](#imaging-rounds)).
* Each value must be a list (JSON array) of locus-specific timepoints, again corresponding to entries in the imaging rounds sequence. A locus imaging timepoint may be present in multiple lists, but it may not occur multiple times within the same list. The regional timepoint itself will also be traced, but it (nor anything else other than locus timepoints) may be present in any list.

If this section is present and nonempty, _only the data present will be used_! Any regional or locus imaging timepoint which is not present will not be traced! Note, however, that any data _not_ present here _must_ be present in the [tracing exclusions section](#tracing-exclusions). Otherwise, if this section's nonempty and missing data isn't in the tracing exclusions, there will be an error.

Proper handling of the interaction between this section and the `tracingExclusions` is done [here](https://github.com/gerlichlab/looptrace/blob/ceed0103b3c68a999b1d975f3ac993d4fec81772/src/test/scala/TestImagingRoundsConfigurationExamplesParsability.scala#L106), with some [example config files here](../src/test/resources/TestImagingRoundsConfiguration/LocusGroupingValidation/).

<a href="tracing-exclusions"></a>
### Tracing exclusions section: `tracingExclusions`
This sections specifies which imaging timepoints should be ignored for chromatin fiber tracing, interlocus distance measurements, and generally other downstream analysis. Note that these timepoints will still have image volumes extracted and have a Gaussian fit done, but they won't be written to the "filtered" traces file (and therefore will not consumed by downstream analyses).

__Guidance__
* `tracingExclusions` should generally be specified, and its value should be a list of timepoints of imaging rounds to exclude from tracing, typically the blank / pre-imaging rounds, and regional barcode rounds. The list can't contain any timepoints not seen in the values of the `imagingRounds`.


## Examples
To see some examples, please refer to [the folder for data for tests of this configuration file](../src/test/resources/TestImagingRoundsConfiguration/).
