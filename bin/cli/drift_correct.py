from looptrace.ImageHandler import ImageHandler
from looptrace.Drifter import Drifter
import sys

if __name__ == '__main__':

    H = ImageHandler(sys.argv[1])
    D = Drifter(H)
    D.drift_corr()