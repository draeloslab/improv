import os
import glob
import caiman as cm

if __name__ == '__main__':
    os.mkdir("./data")
    fld = '/mnt/data/datasets/owen_25_09_03'
    fls = glob.glob(os.path.join(fld,'*.tif'))
    fls.sort()
    m = cm.load_movie_chain(fls[:5])
    split_i = 500
    m[:split_i].save("./data/a.hdf5")
    m[split_i:].save("./data/b.hdf5")
