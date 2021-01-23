import numpy

import parasweep

# period = 10.3
# inc = period/0.4/2
# windows = numpy.round(numpy.arange(inc, inc*21, inc))
# windows = list(map(int, windows))

windows = list(range(3, 80, 3))

# sim_ids_da = parasweep.run_sweep("julia test_colpitts_da_{sim_id}.jl {sim_id}",
#                                  configs=['test_colpitts_da_{sim_id}.jl'],
#                                  templates=["test_colpitts_template"],
#                                  sweep=parasweep.CartesianSweep({'window': windows,
#                                                                  'da': ['true'],
#                                                                  'inflation': numpy.arange(1.0, 1.5, 0.05)}),
#                                  sweep_id='colpitts_da')

# sim_ids_da.to_netcdf("out_colpitts_da.nc")

sim_ids = parasweep.run_sweep("julia test_colpitts_{sim_id}.jl {sim_id}",
                              configs=['test_colpitts_{sim_id}.jl'],
                              templates=["test_colpitts_template"],
                              sweep=parasweep.CartesianSweep({'window': windows,
                                                              'da': ['false'],
                                                              'inflation': [1.0]}),
                              sweep_id='colpitts')

sim_ids.to_netcdf("out_colpitts.nc")
