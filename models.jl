module Models

using Statistics
using NearestNeighbors

struct Model
   p
   integrator
   tendency
end

function integrate(model::Model)
   model.integrator((t, u)->model.tendency(t, u, model.p))
end

function model_error(; x0, model_true, model_err, integrator, outfreq, Δt, N)
   window = outfreq*Δt
   pts = integrator(model_true, x0, 0.0, N*window, Δt, inplace=false)[1:outfreq:end, :]
   errs = zeros(N, size(pts)[2])
   for i=1:N
      res_true = integrator(model_true, pts[i, :], 0.0, window, Δt)
      res_err = integrator(model_err, pts[i, :], 0.0, window, Δt)
      errs[i, :] = (res_true - res_err)
   end
   return errs#sqrt.(mean(errs, dims=1))
end

function model_error_ssa(; x0, model_true, model_err, integrator, outfreq, Δt, N,
                         tree1, tree2, k, r1, r2, psrm=true)
   window = outfreq*Δt
   pts = integrator(model_true, x0, 0.0, N*window, Δt, inplace=false)[1:outfreq:end, :]
   errs = zeros(N, size(pts)[2])
   for i=1:N
      res_true = integrator(model_true, pts[i, :], 0.0, window, Δt)
      res_err = integrator(model_err, pts[i, :], 0.0, window, Δt)
      i1, d1 = knn(tree1, res_err, k)
      i2, d2 = knn(tree2, res_err, k)
      inc = -sum(r2[i2, :] ./ d2, dims=1)/sum(1 ./ d2) + sum(r1[i1, :] ./ d1, dims=1)/sum(1 ./ d1)
      if psrm
         errs[i, :] = (res_true - (res_err + inc'))
      else
         errs[i, :] = (res_true - res_err)
      end
   end
   return errs#sqrt.(mean(errs, dims=1))
end

err_pct = 0.1

function lorenz(t, u, p)
   du = zeros(3)
   du[1] = p["σ"]*(u[2]-u[1])
   du[2] = u[1]*(p["ρ"]-u[3]) - u[2]
   du[3] = u[1]*u[2] - p["β"]*u[3]
   return du
end

lorenz_true = (t, u)->lorenz(t, u, Dict("σ" => 10, "β" => 8/3, "ρ" => 28))
lorenz_err = (t, u)->lorenz(t, u, Dict("σ" => 10.1, "β" => 8/3, "ρ" => 28))

function peña(t, u, p)
   x, y, z, X, Y, Z = u

   du = zeros(6)
   du[1] = p["σ"]*(y - x) - p["c"]*(p["S"]*X + p["k1"])
   du[2] = p["r"]*x - y - x*z + p["c"]*(p["S"]*Y + p["k1"])
   du[3] = x*y - p["b"]*z + p["c_z"]*Z

   du[4] = p["τ"]*p["σ"]*(Y - X) - p["c"]*(x + p["k1"])
   du[5] = p["τ"]*p["r"]*X - p["τ"]*Y - p["τ"]*p["S"]*X*Z + p["c"]*(y + p["k1"])
   du[6] = p["τ"]*p["S"]*X*Y - p["τ"]*p["b"]*Z - p["c_z"]*z

   return du
end

peña_true = (t, u)->peña(t, u, Dict("σ" => 10, "b" => 8/3, "r" => 28,
                                    "c" => 0.15, "c_z" => 0, "k1" => 10,
                                    "S" => 1, "τ" => 0.1))

peña_err = (t, u)->peña(t, u, Dict("σ" => 10 + 0.1, "b" => 8/3, "r" => 28 + 0.1,
                                    "c" => 0.15 - 0.02, "c_z" => 0, "k1" => 10,
                                    "S" => 1, "τ" => 0.1))

function ferrari(t, u, p)
   X, Y, Z, P, Q, ψ_r, ψ_i = u

   F = p["F_0"] + p["F_1"]*cos(p["ω_1"]*t)# + p["F_2"]*cos(p["ω_2"]*(t - p["ϕ"]))
   du = zeros(7)
   du[1] = -(Y^2 + Z^2) - p["a"]*X + p["a"]*F + p["r"]*p["f"]*(P - X - p["γ"])
   du[2] = X*Y - p["b"]*X*Z - Y + p["G"] + p["r"]*p["f"]*(Q - Y)
   du[3] = X*Z + p["b"]*X*Y - Z
   du[4] = -(ψ_r^2 + ψ_i^2)*P + p["f"]/p["c"]*(X - P + p["γ"])
   du[5] = p["f"]/p["c"]*(Y - Q)
   du[6] = -p["σ"]*ψ_r - p["Ω"]*ψ_i + p["α_r"]*X + p["β_r"]*Y
   du[7] = p["Ω"]*ψ_r - p["σ"]*ψ_i + p["α_i"]*X + p["β_i"]*Y

   return du
end

ξ = 8.4e-4
ω_1 = 2*pi/73
ω_2 = ω_1*(40/365)
ferrari_true = (t, u)->ferrari(t, u, Dict("a" => 0.025, "b" => 4, "F_0" => 58.5,
                                          "F_1" => 19.5, "F_2" => 0, "G" => 1,
                                          "f" => 1, "r" => 0.23, "c" => 200,
                                          "γ" => 0.55, "Ω" => 0.0162,
                                          "σ" => 0.00081, "ξ" => ξ, "α_r" => ξ,
                                          "α_i" => ξ/2, "β_r" => ξ/2,
                                          "β_i" => ξ/10, "ω_1" => ω_1,
                                          "ω_2" => ω_2, "ϕ" => 0.4*ω_2))

ferrari_err = (t, u)->ferrari(t, u,
                              Dict("a" => 0.025, "b" => 4, "F_0" => 58.5 + 0.5,
                                   "F_1" => 19.5, "F_2" => 0, "G" => 1,
                                   "f" => 1, "r" => 0.23, "c" => 200,
                                   "γ" => 0.55 - 0.05, "Ω" => 0.0162 + 0.005,
                                   "σ" => 0.00081, "ξ" => ξ, "α_r" => ξ,
                                   "α_i" => ξ/2, "β_r" => ξ/2,
                                   "β_i" => ξ/10, "ω_1" => ω_1 - 0.1,
                                   "ω_2" => ω_2 + 0.1, "ϕ" => 0.2*ω_2))

function rossler(t, u, p)
   n = 2
   du = zeros(3*n)
   for j=1:n
      x, y, z = u[(j - 1)*3 + 1:(j - 1)*3 + 3]
      ω = p["ω_0"] + 0.7*(j - 1)
      du[(j - 1)*3 + 1] = -ω*y - z
      if j == 1
         ym1 = y
      else
         ym1 = u[(j - 2)*3 + 2]
      end
      if j == n
         yp1 = y
      else
         yp1 = u[j*3 + 2]
      end
      du[(j - 1)*3 + 2] = ω*x + p["α"]*y + p["c"]*(yp1 - 2*y + ym1)
      du[(j - 1)*3 + 3] = 0.4 + z*(x - 8.5)
   end
   return du
end

rossler_true = (t, u)->rossler(t, u, Dict("α" => 0.15, "c" => 0.003,
                                          "ω_0" => 1))

rossler_err = (t, u)->rossler(t, u, Dict("α" => 0.16, "c" => 0.004,
                                         "ω_0" => 1 - 0.05))

function colpitts(t, u, p)
   M = 1

   c = [p["c21"], p["c32"], p["c13"]]

   du = zeros(3*M)

   for i=0:M-1
      x1, x2, x3 = u[i*3 + 1:i*3 + 3]
      du[3*i + 1] = p["p1"]*x2 + (c[i%3 + 1])*(u[(3*(i + 1) + 1) % (3*M)] - x1)
      du[3*i + 2] = -p["p2"]*(x1 + x3) - p["p4"]*x2
      du[3*i + 3] = p["p3"][i%3 + 1]*(x2 + 1 - exp(-x1))
   end
   return du
end

colpitts_true = (t, u)->colpitts(t, u, Dict("p1" => 5.0, "p2" => 0.0797,
                                            "p3" => [3.0, 3.5, 4.0],
                                            "p4" => 0.6898, "c21" => 0.05,
                                            "c32" => 0.1, "c13" => 0.15))

colpitts_err = (t, u)->colpitts(t, u, Dict("p1" => 5.0 + 0.1,
                                           "p2" => 0.0797 + 0.01,
                                           "p3" => [3.0, 3.5, 4.0],
                                           "p4" => 0.6898, "c21" => 0.05,
                                           "c32" => 0.1, "c13" => 0.15))

function chua(t, u, p)
   x, y, z = u

   du = zeros(3)

   f = p["m_1"]*x + 0.5*(p["m_0"] - p["m_1"])*(abs(x + 1) - abs(x - 1))

   du[1] = p["α"]*(y - x - f)
   du[2] = x - y + z
   du[3] = -p["β"]*y

   return du
end

chua_true = (t, u)->chua(t, u, Dict("α" => 15.6, "β" => 25.58, "m_1" => -5/7,
                                    "m_0" => -8/7))

chua_err = (t, u)->chua(t, u, Dict("α" => 15.7, "β" => 24.58, "m_1" => -5/7,
                                    "m_0" => -8/7))

function lorenz96(t, u, p)
   N = 36

   # compute state derivatives
   du = zeros(N)

   # first the 3 edge cases: i=1,2,N
   du[1] = (u[2] - u[N-1])*u[N] - u[1]
   du[2] = (u[3] - u[N])*u[1] - u[2]
   du[N] = (u[1] - u[N-2])*u[N-1] - u[N]

   # then the general case
   for i=3:N-1
       du[i] = (u[i+1] - u[i-2])*u[i-1] - u[i]
    end

   du .+= p["F"]

   return du
end

lorenz96_true = (t, u)->lorenz96(t, u, Dict("F" => 8))
lorenz96_err = (t, u)->lorenz96(t, u, Dict("F" => 8.1))

end
