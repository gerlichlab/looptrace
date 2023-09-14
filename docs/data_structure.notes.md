# Notes on `looptrace` data structuring and relationships

## Files

### `*_dc_rois.csv`
* Produced by "spot bounding" / `extract_spots_table.workflow`
* Much bigger than the simple `*_rois.csv` file, since that appears to be for regional spots (without frame), while this file seemingly records a row for each of the ROIs in that file, for each of the frames in the experiment.
* Row count appears to be (very closely?) the product of multiplying the number of frames by the number of rows in the `*_rois.csv` file

### Spot images `.npz` file
* Created by the spot extraction cluster cleanup step
* Read with `np.load(filepath, allow_pickle=True)`
* Result of read is a special data type that we can wrap with `NPZ_wrapper` class for better usage patterns
* Result of read contains a list of files (`.files`) that should match the unique combinations of field-of-view ("position") and "roi_id" from the `*_dc_rois.csv` file, as these are what are zipped up in the spot extraction cleanup step.

