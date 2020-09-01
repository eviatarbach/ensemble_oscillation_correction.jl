using Plots
using DifferentialEquations
using ReservoirComputing
using Statistics

err_pct = 0.05

#lorenz system parameters
u0 = randn(3)#[1.0,0.0,0.0]
tspan = (0.0,2000.0)
p = [10.0,28.0,8/3]
#define lorenz system
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end
#solve and take data
prob = ODEProblem(lorenz, u0, tspan, p)
sol = solve(prob, ABM54(), dt=0.02)
v = sol.u
data = Matrix(hcat(v...))
stds = std(data, dims=2)
data += randn(size(data)).*stds*err_pct
shift = 500
train_len = 98250
predict_len = 1250
train = data[:, shift:shift+train_len-1]
test = data[:, shift+train_len:shift+train_len+predict_len-1]

approx_res_size = 300
radius = 1.2
degree = 6
activation = tanh
sigma = 0.1
beta = 0.5
alpha = 1.0
nla_type = NLAT2()
extended_states = false

esn = ESN(approx_res_size,
    train,
    degree,
    radius,
    activation = activation, #default = tanh
    alpha = alpha, #default = 1.0
    sigma = sigma, #default = 0.1
    nla_type = nla_type, #default = NLADefault()
    extended_states = extended_states #default = false
    )

W_out = ESNtrain(esn, beta)
output = ESNpredict(esn, predict_len, W_out)

#plot(transpose(output),layout=(3,1), label="predicted")
#plot!(transpose(test),layout=(3,1), label="actual")

# Resynchronize

esn2 = ESN(esn.W,
           test[:, 850:900], esn.W_in,
           activation = activation, #default = tanh
           alpha = alpha, #default = 1.0
           nla_type = nla_type, #default = NLADefault()
           extended_states = extended_states) #default = false

ESNtrain(esn2, beta)
output2 = ESNpredict(esn2, predict_len, W_out)

plot(transpose(output2),layout=(3,1), label="predicted")
plot!(transpose(test[:, 900:end]),layout=(3,1), label="actual")
