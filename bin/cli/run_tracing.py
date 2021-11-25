from looptrace import ImageHandler, Tracer
import sys

if __name__ == '__main__':
    H = ImageHandler(sys.argv[1])
    T = Tracer(H)
    T.make_dc_rois_all_frames()
    T.tracing_3d()