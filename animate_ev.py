import matplotlib.pyplot as plt
import xarray
import numpy

mode = 3
D = 4964
M = 60

prec = xarray.open_dataset('/lustre/ebach/imd/prec.nc')
mask = ~numpy.isnan(prec["p"][0, :, :]).values
lons, lats = numpy.where(mask)

eig_vecs = xarray.open_dataset('eig_vecs.h5')

ev = eig_vecs.eig_vecs[mode, :].values
ev = numpy.reshape(ev, (D, M))

def animate(i):
    plt.clf()
    m = numpy.nan*numpy.zeros(mask.shape)
    m[lons, lats] = ev[:, i]
    plt.contourf(m, 40, vmin=0, vmax=vmax)
    plt.savefig('anim_' + str(i).zfill(2) + '.png')

for i in range(M):
    animate(i)
