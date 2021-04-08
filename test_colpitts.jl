include("models.jl")
include("integrators.jl")
include("enoc.jl")
using .Models
using .Integrators
using .enoc

M = 30
D = 6
k = 40
k_r = 30

osc_vars = 1:D
modes = 2:3
model = Models.colpitts_true
model_err = Models.colpitts_err
integrator = Integrators.rk4
outfreq = 4
Δt = 0.1
m = 20
cycles = 100
window = 40
record_length = 25000
ens_err_pct = 0.2
obs_err_pct = 0.1
varimax = false
transient = 3000
mp = 9
da = false
inflation = false
y_fcst = false

y0 = rand(D)

info, ssa_info = enoc.run(model=model, model_err=model_err, M=M, D=D, k=k,
                          k_r=k_r, modes=modes, osc_vars=osc_vars,
                          integrator=integrator, outfreq=outfreq, Δt=Δt, m=m,
                          cycles=cycles, window=window,
                          record_length=record_length, ens_err_pct=ens_err_pct,
                          obs_err_pct=obs_err_pct, transient=transient, y0=y0,
                          mp=mp, varimax=varimax, check_bounds=false,
                          test_time=nothing, da=da, inflation=inflation,
                          y_fcst=false, preload="colpitts")
