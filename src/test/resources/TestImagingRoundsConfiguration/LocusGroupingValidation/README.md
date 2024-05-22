### `LocusGroupingValidation` tests
Here, some files are for asserting that no regional timepoint may appear in any list of values in the `locusGrouping` section; those values lists may only contain non-regional rounds' timepoints.

Some other files here are for tests of [Issue 304](https://github.com/gerlichlab/looptrace/issues/304).
We test the behavior of the locus grouping validation step which says that in general, any locus imaging timepoint _not_ in the `tracingExclusions` section of the imaging rounds config _must_ be in at least one of the values lists in the `locusGrouping` section (assuming that section's present). Note that this _does not apply to non-locus imaging timepoints_. In particular, a rounds tagged as `isBlank` (set to `true`) is not subject to this restriction and may be omitted from the `locusGrouping` even if not present in the `tracingExclusions`.

Finally, still other files here are for validating behavior when toggling-off the checking that the union of values in the `locusGrouping` must cover the set of locus imaging timepoints from the `imagingRounds`. This corresponds to [Issue 295](https://github.com/gerlichlab/looptrace/issues/295) and is a manual, more universal version of the narrower, automatic behavior described in [Issue 304](https://github.com/gerlichlab/looptrace/issues/304).
