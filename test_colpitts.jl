#using Distributed

#rmprocs(procs())
#addprocs()

include("models.jl")
include("integrators.jl")
include("run_da_ssa.jl")
using .Models
using .Integrators
using .run_da_ssa

M = 30
D = 9
k = 41

osc_vars = 1:D
modes = 1:4
model = Models.colpitts_true
model_err = Models.colpitts_err
integrator = Integrators.rk4
outfreq = 4
Δt = 0.1
m = 10
cycles = 100
window = 20
inflation1 = 1.3
inflation2 = 1.3
record_length = 10000.0
obs_err_pct = 0.1
ens_err_pct = 0.2
transient = 500
cov = false

info1, info2, ssa_info = run_da_ssa.etkf_da_ssa_compare(model=model, model_err=model_err,
                                              M=M, D=D, k=k, modes=modes,
                                              osc_vars=osc_vars,
                                              integrator=integrator,
                                              outfreq=outfreq, Δt=Δt,
                                              m=m, cycles=cycles, window=window,
                                              inflation1=inflation1, inflation2=inflation2,
                                              record_length=record_length,
                                              obs_err_pct=obs_err_pct,
                                              ens_err_pct=ens_err_pct,
                                              transient=transient, cov=cov)
