import datetime

import matplotlib.pyplot as plt
import xarray
import numpy
import matplotlib.animation as animation

mode = 3
D = 4964
M = 60
year = 1910
year0 = 1901

date = datetime.datetime(month=5, day=1, year=year)

mask = xarray.open_dataset('mask.h5')["mask"].values
lons, lats = numpy.where(mask)

r = xarray.open_dataset('r_summed.h5')["r_summed"]

ev = r[year-year0-1, :, :].values
vmin = -10
vmax = 10
levels = numpy.linspace(vmin, vmax, 41)

def animate(i):
    global date
    ax.clear()
    m = 4*numpy.ones(mask.shape)
    m[lons, lats] = ev[:, i]
    ax.contourf(numpy.arange(66.5, 100.25, 0.25),
		numpy.arange(6.5, 38.75, 0.25), m, levels=levels,
		extend='both', cmap='RdBu')
    ax.set_title(date.strftime("%d %B %Y"))
    date += datetime.timedelta(days=1)
    return ax
    #plt.savefig('anim_' + str(i).zfill(2) + '.png')

fig = plt.figure()
ax = fig.gca()

m = 4*numpy.ones(mask.shape)
m[lons, lats] = ev[:, 0]
contourplot = ax.contourf(numpy.arange(66.5, 100.25, 0.25),
		          numpy.arange(6.5, 38.75, 0.25), m, levels=levels,
			  extend='both', cmap='RdBu')
cbar = plt.colorbar(contourplot)

ani = animation.FuncAnimation(fig, animate, frames=range(153))
ani.save('movie2.gif', writer='imagemagick')
#for i in range(M):
#    animate(i)
