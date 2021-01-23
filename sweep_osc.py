import numpy

import parasweep

period = 2.8
inc = period/0.5/2
windows = numpy.round(numpy.arange(inc, inc*31, inc))
windows = list(map(int, windows))

sim_ids_da = parasweep.run_sweep("julia test_osc_da_{sim_id}.jl {sim_id}",
                                 configs=['test_osc_da_{sim_id}.jl'],
                                 templates=["test_osc_template"],
                                 sweep=parasweep.CartesianSweep({'window': windows,
                                                                 'da': ['true'],
                                                                 'inflation': numpy.arange(1.0, 1.5, 0.05)}),
                                 sweep_id='osc_da')

sim_ids_da.to_netcdf("out_osc_da.nc")

sim_ids = parasweep.run_sweep("julia test_osc_{sim_id}.jl {sim_id}",
                              configs=['test_osc_{sim_id}.jl'],
                              templates=["test_osc_template"],
                              sweep=parasweep.CartesianSweep({'window': windows,
                                                              'da': ['false'],
                                                              'inflation': [1.0]}),
                              sweep_id='osc')

sim_ids.to_netcdf("out_osc.nc")
