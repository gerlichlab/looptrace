# `looptrace` imaging rounds configuration file
The imaging rounds configuration file is where you tell `looptrace` about the rounds of imaging you did during a sequential FISH experiment.
It's from this file that a name becomes associated with each imaging round, rounds are conceptually distinguished from one another (e.g., blank vs. regional or local FISH), and the relationships between rounds are defined for analysis.

## Requirements and suggestions
* `regionGrouping` is required and should be a mapping, with at minimum a `semantic` key. The value for the semantic may be "Trivial", "Permissive", or "Prohibitive", depending on how you want the groupings to be interpreted with respect to excluding proximal spots. 
"Trivial" treats all regional spots as one big group and will exclude any that violate the proximity threshold; it's like "Prohibitive" but with no grouping.
"Permissive" allows spots from timepoints grouped together to violate the proximity threshold; spots from timepoints not grouped together may not violate the proximity threshold.
"Prohibitive" forbids spots from timepoints grouped together to violate the proximity threshold; spots from timepoints not grouped together may violate the proximity threshold.
* If `semantic` is set to "Trivial", there can be no `groups`; otherwise, `groups` is required and must be an array-of-arrays.
* `semantic` and `groups` must be nested within the `regionGrouping` mapping.
* If specified, the value for `groups` must satisfy these properties:
    * Each group must have no repeat value.
    * The groups must be disjoint.
    * The union of the groups must cover the set of regional round timepoints from the `imagingRounds`.
    * The set of regional round timepoints from the `imagingRounds` must cover the union of groups.
* `minimumPixelLikeSeparation` must be a key in the `regionGrouping` mapping and should be a nonnegative integer. 
This represents the minimum separation (in pixels) in each dimension between the centroids of regional spots. 
NB: For z, this is slices not pixels.
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
* Check that the list of `tracingExclusions` values is correct, most likely any pre-imaging timepoint names, "blank" timepoints, and all regional barcode timepoint names.
* `tracingExclusions` should generally be specified, and its value should be a list of timepoints of imaging rounds to exclude from tracing, typically the blank / pre-imaging rounds, and regional barcode rounds. The list can't contain any timepoints not seen in the values of the `imagingRounds`.
