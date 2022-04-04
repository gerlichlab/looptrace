from looptrace.ImageHandler import ImageHandler
from looptrace.Tracer import Tracer
from looptrace.SpotPicker import SpotPicker
import sys

if __name__ == '__main__':
    H = ImageHandler(sys.argv[1])
    S = SpotPicker(H)
    S.rois_from_beads()
    T = Tracer(H, trace_beads=True)
    T.make_dc_rois_all_frames()
    T.tracing_3d()
    T = Tracer(H)
    T.make_dc_rois_all_frames()
    T.tracing_3d()