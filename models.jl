module Models

using Statistics
using NearestNeighbors
using StaticArrays

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

function colpitts(t, u, p)
   M = 2

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
                                            "p3" => 3*[3.0, 3.5, 4.0],
                                            "p4" => 0.6898, "c21" => 0.05,
                                            "c32" => 0.1, "c13" => 0.15))

colpitts_err = (t, u)->colpitts(t, u, Dict("p1" => 5.0 + 0.1,
                                           "p2" => 0.0797 + 0.01,
                                           "p3" => 3*[3.0, 3.5, 4.0],
                                           "p4" => 0.6898, "c21" => 0.05,
                                           "c32" => 0.1, "c13" => 0.15))

function duffing!(du, t, u, p)
   x, y = u
   #du = zeros(2)

   du[1] = y
   du[2] = p["a"]*cos(p["ω"]*t)*x - x^3 - p["b"]*y

   return du
end

duffing_true = (du, t, u)->duffing!(du, t, u, Dict("ω" => 0.7, "a" => 6.25, "b" => 0.3))

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
   N = 9

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

function forced!(du, t, u, p)
   x, y, uu, v = u

   #du = zeros(4)

   du[1] = v
   du[2] = uu
   du[3] = -p["Ω"]^2*y
   du[4] = -sin(x) + y

   return du
end

forced_true = (du, t, u)->forced!(du, t, u, Dict("Ω" => 1))

function osc(t, u, p)
   x, y, z, uu, v = u

   du = zeros(5)
   du[1] = p["σ"]*(u[2]-u[1]) + p["c"]*uu
   du[2] = u[1]*(p["ρ"]-u[3]) - u[2]
   du[3] = u[1]*u[2] - p["β"]*u[3]
   du[4] = v
   du[5] = -p["Ω"]^2*uu

   return du
end

osc_true = (t, u)->osc(t, u, Dict("Ω" => 0.3, "σ" => 10, "β" => 8/3,
                                  "ρ" => 28, "c" => 5))

osc_err = (t, u)->osc(t, u, Dict("Ω" => 0.32, "σ" => 10, "β" => 8/3,
                                 "ρ" => 28, "c" => 5.1))

function ly!(du, t, u, p)
   θ, y = u
   #du = zeros(2)

   Λ = p["k_1"]*sin(p["ω_1"]*t) + p["k_2"]*sin(p["ω_2"]*t)
   du[1] = 2*pi/p["P"] + p["σ"]*y
   du[2] = -p["λ"]*y + p["γ"]*sin(θ)*Λ

   return du
end

ly_true = (du, t, u)->ly!(du, t, u, Dict("ω_1" => 2*pi/41, "ω_2" => 2*pi/23,
                                         "k_1" => 0.8, "k_2" => 0.8, "P" => 100,
                                         "σ" => 3, "λ" => 0.2, "γ" => 0.1))

function osc84!(du, t, u, p)
  x, y, z = u
  #du = zeros(3)

  du[1] = -p["a"]*x - y^2 - z^2 + p["a"]*p["F"]
  du[2] = -y + x*y - p["b"]*x*z + p["G"]
  du[3] = -z + p["b"]*x*y + x*z + p["ϵ"]*cos(p["ω"]*t)

  return du
end

osc84_true = (du, t, u)->osc84!(du, t, u, Dict("a" => 1/4, "b" => 4, "ϵ" => 3,
                                               "ω" => 2*pi/73, "F" => 11,
                                               "G" => 0.5))


end
