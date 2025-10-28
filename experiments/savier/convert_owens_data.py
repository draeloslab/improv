import os
import glob
import caiman as cm
fld = '/mnt/data/datasets/owen_25_09_03'  # path to folder where the data is located
fls = glob.glob(os.path.join(fld,'*.tif'))  #  change tif to the extension you need
fls.sort()  # make sure your files are sorted alphanumerically
m = cm.load_movie_chain(fls[:5])
split_i = 500
m[:split_i].save("./output/a.hdf5")
m[split_i:].save("./data/b.hdf5")


# m.save(os.path.join(fld,'data.tif'))
