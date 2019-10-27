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
k = 5

osc_vars = 1:D
modes = 2:3
model = Models.colpitts_true
model_err = Models.colpitts_err
integrator = Integrators.rk4
outfreq = 10
Δt = 0.1
m = 20
cycles = 100
window = 10*outfreq*Δt
inflation1 = 1.1
inflation2 = 1.1
record_length = 10000.0
obs_err_pct = 0.1
ens_err_pct = 0.01
transient = 500
cov = false

info1, info2, ssa_info1, ssa_info2 = run_da_ssa.etkf_da_ssa_compare(model=model, model_err=model_err,
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
