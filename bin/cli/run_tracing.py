from looptrace.ImageHandler import ImageHandler
from looptrace.Tracer import Tracer
from looptrace.SpotPicker import SpotPicker
import sys

if __name__ == '__main__':
    #H = ImageHandler(sys.argv[1])
    #S = SpotPicker(H)
    #S.rois_from_spots(filter_nucs=False)
    H = ImageHandler(sys.argv[1])
    T = Tracer(H)
    T.make_dc_rois_all_frames()
    #T.gen_roi_imgs_inmem_coursedc()
    T.trace_all_rois()