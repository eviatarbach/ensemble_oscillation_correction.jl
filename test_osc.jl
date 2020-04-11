#using Distributed

#rmprocs(procs())
#addprocs()

include("models.jl")
include("integrators.jl")
include("run_ens_forecast.jl")
using .Models
using .Integrators
using .run_ens_forecast

M = 100
D = 5
k = 40
k_r = 30

osc_vars = 1:2
modes = 1:2
model = Models.osc_true
model_err = Models.osc_err
integrator = Integrators.rk4
outfreq = 10
Δt = 0.05
m = 20
cycles = 100
window = 70
record_length = 10000.0
ens_err_pct = 0.2
obs_err_pct = 0.1
brownian_noise = false
varimax = false
transient = 2000
mp = 9

y0 = [randn(3)..., 0, 0.3*10]

info, ssa_info = run_ens_forecast.ens_forecast_compare(model=model, model_err=model_err,
                                              M=M, D=D, k=k, k_r=k_r, modes=modes,
                                              osc_vars=osc_vars,
                                              integrator=integrator,
                                              outfreq=outfreq, Δt=Δt,
                                              m=m, cycles=cycles, window=window,
                                              record_length=record_length,
                                              ens_err_pct=ens_err_pct, obs_err_pct=obs_err_pct,
                                              transient=transient, brownian_noise=brownian_noise,
                                              y0=y0, mp=mp, varimax=varimax)