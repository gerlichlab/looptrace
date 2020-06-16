# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:46:38 2020

@author: ellenberg
"""

from chromatin_tracing_python.tracer_h5 import Tracer_decon
from tkinter.filedialog import askopenfilename
import os


def main():
    config_path = askopenfilename(title = "Select tracing configuration file")
    T = Tracer_decon(config_path)
    print('Tracer initialized with config: ', config_path, T.config)
    traces, imgs = T.tracing_multi_decon()
    print('Tracing complete, saving data.')
    T.save_data(traces=traces, imgs=imgs, config=T.config)
    print('Data saved in output folder.')
    
if __name__ == '__main__':
    main()