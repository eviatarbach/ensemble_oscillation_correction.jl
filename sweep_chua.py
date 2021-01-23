import numpy

import parasweep

#period = 2.1
#inc = period/0.1/2
#windows = numpy.round(numpy.arange(inc, inc*21, inc))
#windows = list(map(int, windows))
windows = list(range(3, 85, 3))

#sim_ids_da = parasweep.run_sweep("julia test_chua_da_{sim_id}.jl {sim_id}",
#                                 configs=['test_chua_da_{sim_id}.jl'],
#                                 templates=["test_chua_template"],
#                                 sweep=parasweep.CartesianSweep({'window': windows,
#                                                                 'da': ['true'],
#                                                                 'inflation': numpy.arange(1.0, 1.5, 0.05)}),
#                                 sweep_id='chua_da_new')

#sim_ids_da.to_netcdf("out_chua_da.nc")

sim_ids = parasweep.run_sweep("julia test_chua_{sim_id}.jl {sim_id}",
                              configs=['test_chua_{sim_id}.jl'],
                              templates=["test_chua_template"],
                              sweep=parasweep.CartesianSweep({'window': windows,
                                                              'da': ['false'],
                                                              'inflation': [1.0]}),
                              sweep_id='chua_new')

#sim_ids.to_netcdf("out_chua.nc")
