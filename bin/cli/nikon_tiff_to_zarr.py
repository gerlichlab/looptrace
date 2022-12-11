"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""


import glob
import re
from joblib import Parallel, delayed
from numcodecs import Blosc
import zarr
import tqdm
import os
import numpy as np
import tifffile
import sys

class FileConverter():
    def __init__(self, input_folder, image_path):
        self.input_folder = input_folder
        self.image_path = image_path

    def gen_folder_list(self):
        
        folders = sorted([self.input_folder+os.sep+folder for folder in os.listdir(self.input_folder)])

        fnarr = []
        t_max = 0
        for i, folder in enumerate(folders):
            files = sorted(glob.glob(folder+os.sep+'*.tiff'))
            print(files[0])
            a = np.empty((len(files), 4)).astype(object)

            for j, f in enumerate(files):
                match = re.findall('(Time)(\d+)_(Point)(\d+)_(ZStack)(\d+)', f)
                _, t, _, p, _, z = match[0]
                t_new = t_max + int(t)
                a[j] = ([int(p), t_new, int(z), f])
            t_max = t_new + 1
            fnarr.append(a)
        fnarr = np.concatenate(fnarr, axis=0)
        fnarr = fnarr[fnarr[:,3].argsort()]
        self.fnarr = fnarr
        np.savetxt(self.image_path+os.sep+'fnarr.txt', fnarr, fmt='%s')

    def read_stack(self, p, t):
        fnarr = self.fnarr
        fns = fnarr[(fnarr[:,0] == p) & (fnarr[:,1] == t)][:,3]
        img = np.moveaxis(np.stack([tifffile.imread(fn) for fn in fns]), 0, 1)
        with open(self.image_path+os.sep+"indexes.txt", "a") as f:
            f.write(str(['Indexes, num files: ', p, t, len(fns)]))
            f.write('\n')
            f.write(str(['Image shape: ', img.shape]))
            f.write('\n')
            np.savetxt(f, fns, fmt='%s')
        return img

    def single_image_to_zarr(self, store, pos, t):
        stack = self.read_stack(pos, t)
        try:
            store[t] = stack
        except ValueError:
            shape_diff = tuple(np.array(store.shape[1:])-np.array(stack.shape))
            pad = tuple((0, i) for i in shape_diff)
            store[t] = np.pad(stack, pad)

    def run_conversion(self, continue_from = None):

        sample = tifffile.imread(self.fnarr[0,3])
        C, Y, X = sample.shape
        P, T, Z = np.max(self.fnarr[:,0:3], axis=0)+1
        dt = np.uint16

        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        chunks = (1,1,1,Y//2,X//2)

        if not os.path.isdir(self.image_path):
            os.makedirs(self.image_path)

        
        pos_list = ['P'+str(p+1).zfill(4) for p in range(P)]
        if continue_from is not None:
            print('Continuing from ', continue_from)
            cont_index = pos_list.index(continue_from)
            pos_list = pos_list[cont_index:]
            print('Remaining positions: ', pos_list)

        for pos in tqdm.tqdm(pos_list):
            pos_index = pos_list.index(pos)

            store = zarr.DirectoryStore(self.image_path+os.sep+pos)
            root = zarr.group(store=store, overwrite=True)

            root.attrs['multiscale'] = {'multiscales': [{'version': '0.3', 
                                                'name': pos, 
                                                'datasets': [{'path': '0'}],
                                                'axes': ['t','c','z','y','x']}]}

            multiscale_level = root.create_dataset(name = str(0), compressor=compressor, shape=(T, C, Z, Y, X), chunks=chunks, dtype = dt)

            Parallel(n_jobs=-1, prefer='threads', verbose=10)(delayed(self.single_image_to_zarr)(multiscale_level, pos_index, t_idx) for t_idx in range(T))


if __name__ == '__main__':

    input_folder = sys.argv[1]
    image_path = sys.argv[2]

    F = FileConverter(input_folder, image_path)
    F.gen_folder_list()
    F.run_conversion()
