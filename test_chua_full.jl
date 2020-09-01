include("models.jl")
include("integrators.jl")
include("full.jl")
using .Models
using .Integrators
using .full

D = 3
k = 40

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
transient = 3000
inflation = 1.05

y0 = [-1.06095, 0.160678, 0.267729]

info = full.run(model=model, model_err=model_err, D=D, k=k,
                  integrator=integrator, outfreq=outfreq, Δt=Δt, m=m,
                  cycles=cycles, window=window,
                  record_length=record_length, ens_err_pct=ens_err_pct,
                  obs_err_pct=obs_err_pct, transient=transient, y0=y0,
                  check_bounds=true,
                  test_time=10.0, inflation=inflation)
