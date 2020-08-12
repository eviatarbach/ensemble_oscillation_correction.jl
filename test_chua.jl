#using Distributed

#rmprocs(procs())
#addprocs()

include("models.jl")
include("integrators.jl")
include("enoc.jl")
using .Models
using .Integrators
using .enoc

M = 60
D = 3
k = 40
k_r = 30

osc_vars = 1:D
modes = 3:4
model = Models.chua_true
model_err = Models.chua_err
integrator = Integrators.rk4
outfreq = 1
Δt = 0.1
m = 20
cycles = 100
window = 10
record_length = 25000
ens_err_pct = 0.2
obs_err_pct = 0.1
brownian_noise = false
transient = 3000
mp = 9
varimax = true
da = false
inflation = false

y0 = [-1.06095, 0.160678, 0.267729]

info, ssa_info = enoc.run(model=model, model_err=model_err, M=M, D=D, k=k,
                          k_r=k_r, modes=modes, osc_vars=osc_vars,
                          integrator=integrator, outfreq=outfreq, Δt=Δt, m=m,
                          cycles=cycles, window=window,
                          record_length=record_length, ens_err_pct=ens_err_pct,
                          obs_err_pct=obs_err_pct, transient=transient, y0=y0,
                          mp=mp, varimax=varimax, check_bounds=true,
                          test_time=10.0, da=da, inflation=inflation)
