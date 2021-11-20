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

if __name__ == '__main__':

    input_folder = sys.argv[1]
    image_path = sys.argv[2]

    folders = [input_folder+os.sep+folder for folder in os.listdir(input_folder)]

    fnarr = []

    for i, folder in enumerate(folders):
        t_max = 0
        if i>0:
            t_max = int(t)+1

        files = glob.glob(folder+os.sep+'*.tiff')
        print(files[0])
        a = np.empty((len(files), 4)).astype(object)

        for i, f in enumerate(files):
            match = re.findall('(Time)(\d+)_(Point)(\d+)_(ZStack)(\d+)', f)
            _, t, _, p, _, z = match[0]
            t = int(t)+t_max
            a[i] = ([int(p), int(t), int(z), f])
        fnarr.append(a)
    fnarr = np.concatenate(fnarr, axis=0)
    fnarr = fnarr[fnarr[:,3].argsort()]
    print(fnarr.shape)
    print(fnarr[0])
    sample = tifffile.imread(files[0])
    C, Y, X = sample.shape
    P, T, Z = np.max(a[:,0:3], axis=0)+1
    dt = sample.dtype

    def read_stack(fnarr, p, t):
        cond = np.where((fnarr[:,0] == p) & (fnarr[:,1] == t))
        fns = fnarr[cond][:,3]
        img = np.moveaxis(np.stack([tifffile.imread(fn) for fn in fns]), 0, 1)
        print(img.shape)
        return img

    def single_image_to_zarr(store, fnarr, pos, t):
        store[t] = read_stack(fnarr, pos, t)

    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    chunks = (1,1,1,Y//2,X//2)



    if not os.path.isdir(image_path):
        os.mkdir(image_path)

    pos_list = ['P'+str(p+1).zfill(4) for p in range(P)]

    for pos in tqdm.tqdm(pos_list):
        pos_index = pos_list.index(pos)

        store = zarr.DirectoryStore(image_path+os.sep+pos)
        root = zarr.group(store=store, overwrite=True)

        root.attrs['multiscale'] = {'multiscales': [{'version': '0.2', 'name': 'dataset', 'datasets': [{'path': '0'}]}]}

        multiscale_level = root.create_dataset(name = str(0), compressor=compressor, shape=(T, C, Z, Y, X), chunks=chunks)

        Parallel(n_jobs=-1, prefer='threads', verbose=10)(delayed(single_image_to_zarr)(multiscale_level, fnarr, pos_index, t_idx) for t_idx in range(T))

