# `looptrace` signal analysis configuration file
This file is in JSON format and is optional when [running the pipeline](./running-the-pipeline.md). 
If provided, it tells `looptrace` how to analyze signal (e.g., from immunofluorescence) from channels other than the one(s) in which FISH spots were detected, in regions defined by the centroids of the FISH spots.

The format is a list-of-objects. Here are the components of each object:
    * `roiType`: This key maps to which ROI type to consider; this must be either "Regional" (for now, "LocusSpecific" will become supported.)
    * `roiDiameterInPixels`: This key maps to the "diameter" (really the side length of a square) of the 2D region in which to analyze signal, centered around each spot center. This must be a positive integer. $z$ is aggregated in several ways.
    * `channels`: This key maps to which channels in which signal should be analyzed; this must be a list of nonnegative integers.

Here's an example:
```console
[
  {
    "roiType": "Regional",
    "roiDiameterInPixels": 20,
    "channels": [
      1,
      2
    ]
  }
]
```
