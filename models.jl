module Models

using Statistics
using NearestNeighbors

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

end
