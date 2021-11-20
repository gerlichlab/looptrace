from looptrace import ImageHandler
import sys

if __name__ == '__main__':

    H = ImageHandler(sys.argv[1])

    H.dask_to_ome_zarr()