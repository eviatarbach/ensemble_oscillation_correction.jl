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
D = 4
k = 10
k_r = 10

osc_vars = 1:D
modes = 1:2
model = Models.lorenz96_true
model_err = Models.lorenz96_true
integrator = Integrators.rk4
outfreq = 10
Δt = 0.1
m = 20
cycles = 100
window = 40
record_length = 100000
ens_err_pct = 0.2
obs_err_pct = 0.1
brownian_noise = false
transient = 3000
mp = 9
varimax = false

y0 = randn(D)#[0.723667, 0.101699, 0.0719784, 0.923862, 0.321385, 0.579979]

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
