import xarray
import h5py

prec = xarray.open_dataset('/lustre/ebach/imd/prec.nc')
mask = ~numpy.isnan(prec["p"][0, :, :]).values

h5f = h5py.File('mask.h5', 'w')
h5f.create_dataset('mask', data=mask.astype(float))
h5f.close()
