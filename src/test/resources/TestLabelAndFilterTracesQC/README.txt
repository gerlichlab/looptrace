The expectations in the labeled_trace_outfile.txt file are based on the following settings:
1. withinRegion: Maximum distance-to-region of 800 (nanometers)
2. sufficientSNR: Minimum signal-to-noise ratio of 2
3. denseXY: Standard deviation in XY of no more than 150
4. denseZ: Standard deviation in Z of no more than 400
5. inBoundsX: x-coordinate (x) of centroid of fit within (sigmaXY, 32*107.326 - sigmaXY)
6. inBoundsY: y-coordinate (y) of centroid of fit within (sigmaXY, 32*107.326 - sigmaXY)
7. inBoundsZ: z-coordinate (z) of centroid of fit within (sigmaZ, 16*300 - sigmaZ)

The above are settings which were commonly used during testing of looptrace.
The box sizes are based on xy scaling of 107.326078090454419 nanometers per pixel, and 
z scaling of 300 nanometers per pixel, and a box size of (16, 32, 32) for (z, y, x).