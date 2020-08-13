import matplotlib.pyplot as plt
import xarray
import numpy
import matplotlib.animation as animation

mode = 3
D = 4964
M = 60

mask = xarray.open_dataset('mask.h5')["mask"].values
lons, lats = numpy.where(mask)

eig_vecs = xarray.open_dataset('eig_vecs.h5')

ev = eig_vecs.eig_vecs[mode, :].values
ev = numpy.reshape(ev, (D, M))
vmax = numpy.max(ev)
vmin = numpy.min(ev)

def animate(i):
    ax.clear()
    m = numpy.nan*numpy.zeros(mask.shape)
    m[lons, lats] = ev[:, i]
    ax.contourf(m, 40, vmin=vmin, vmax=vmax, cmap='bwr_r')
    return ax
    #plt.savefig('anim_' + str(i).zfill(2) + '.png')

fig = plt.figure()
#fig.colorbar()
ax = fig.gca()
ani = animation.FuncAnimation(fig, animate, frames=range(M))
ani.save('movie.gif', writer='imagemagick')
#for i in range(M):
#    animate(i)
