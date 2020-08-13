import xarray
import numpy

n_years = 116
n_days = 153  # May, June, July, August, September
D = 4964

prec = xarray.open_dataset('prec.nc')

m = prec["p"]["time"].dt.month
mask = (m >= 5) & (m <= 9)
prec_masked = prec["p"][mask]
mask_latlon = ~numpy.isnan(prec["p"][mask][0, :, :])

years = set(prec_masked.time.dt.year.values)

prec_full = numpy.zeros((n_days, D, n_years))

for i, year in enumerate(years):
    prec_full[:, :, i] = prec_masked[prec_masked.time.dt.year == year].values[:, mask_latlon]

prec_full = prec_full - numpy.mean(prec_full)

xarray.DataArray(prec_full, name="prec").to_netcdf('prec_out.nc')
