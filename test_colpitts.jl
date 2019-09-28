using Distributed

rmprocs(procs())
addprocs()

@everywhere include("models.jl")
@everywhere include("integrators.jl")
@everywhere include("run_da_ssa.jl")
@everywhere using .Models
@everywhere using .Integrators
@everywhere using .run_da_ssa

M = 30
D = 9
k = 5

osc_vars = 1:D
modes = 2:3
model = Models.colpitts_true
model_err = Models.colpitts_err
integrator = Integrators.rk4
outfreq = 4
Δt = 0.1
m = 20
cycles = 1000
window = 3.0
inflation1 = 1.2
inflation2 = 1.27
record_length = 10000.0
obs_err_pct = 0.1
ens_err_pct = 0.1
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
