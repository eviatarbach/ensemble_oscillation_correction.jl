module Models

export lorenz, peña, ferrari, rossler

function lorenz(t, u)
 du = zeros(3)
 du[1] = 10.0*(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
 return du
end

function lorenz2(t, u)
 du = zeros(3)
 du[1] = 10.1*(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
 return du
end

function peña(t, u)
   σ = 10
   b = 8/3
   r = 28
   c = 0.15
   c_z = 0
   k1 = 10
   S = 1
   τ = 0.1

   x, y, z, X, Y, Z = u

   du = zeros(6)
   du[1] = σ*(y - x) - c*(S*X + k1)
   du[2] = r*x - y - x*z + c*(S*Y + k1)
   du[3] = x*y - b*z + c_z*Z

   du[4] = τ*σ*(Y - X) - c*(x + k1)
   du[5] = τ*r*X - τ*Y - τ*S*X*Z + c*(y + k1)
   du[6] = τ*S*X*Y - τ*b*Z - c_z*z

   return du
end

function ferrari(t, u)
   a = 0.025
   b = 4
   F_0 = 58.5
   F_1 = 19.5
   F_2 = 0#10
   G = 1
   f = 1
   r = 0.23
   c = 200
   γ = 0.55
   Ω = 0.0162
   σ = 0.00081
   ξ = 8.4e-4
   α_r = ξ
   α_i = ξ/2
   β_r = ξ/2
   β_i = ξ/10
   ω_1 = 2*pi/73
   ω_2 = ω_1*(40/365)
   ϕ = 0.4*ω_2

   X, Y, Z, P, Q, ψ_r, ψ_i = u

   F = F_0 + F_1*cos(ω_1*t) + F_2*cos(ω_2*(t - ϕ))
   du = zeros(7)
   du[1] = -(Y^2 + Z^2) - a*X + a*F + r*f*(P - X - γ)
   du[2] = X*Y - b*X*Z - Y + G + r*f*(Q - Y)
   du[3] = X*Z + b*X*Y - Z
   du[4] = -(ψ_r^2 + ψ_i^2)*P + f/c*(X - P + γ)
   du[5] = f/c*(Y - Q)
   du[6] = -σ*ψ_r - Ω*ψ_i + α_r*X + β_r*Y
   du[7] = Ω*ψ_r - σ*ψ_i + α_i*X + β_i*Y

   return du
end

function ferrari2(t, u)
   a = 0.025
   b = 4
   F_0 = 58.5 + 0.5
   F_1 = 19.5
   F_2 = 0#10
   G = 1
   f = 1
   r = 0.23
   c = 200
   γ = 0.55 - 0.05
   Ω = 0.0162
   σ = 0.00081
   ξ = 8.4e-4
   α_r = ξ
   α_i = ξ/2
   β_r = ξ/2
   β_i = ξ/10
   ω_1 = 2*pi/73
   ω_2 = ω_1*(40/365)
   ϕ = 0.4*ω_2

   X, Y, Z, P, Q, ψ_r, ψ_i = u

   F = F_0 + F_1*cos(ω_1*t) + F_2*cos(ω_2*(t - ϕ))
   du = zeros(7)
   du[1] = -(Y^2 + Z^2) - a*X + a*F + r*f*(P - X - γ)
   du[2] = X*Y - b*X*Z - Y + G + r*f*(Q - Y)
   du[3] = X*Z + b*X*Y - Z
   du[4] = -(ψ_r^2 + ψ_i^2)*P + f/c*(X - P + γ)
   du[5] = f/c*(Y - Q)
   du[6] = -σ*ψ_r - Ω*ψ_i + α_r*X + β_r*Y
   du[7] = Ω*ψ_r - σ*ψ_i + α_i*X + β_i*Y

   return du
end

function rossler(t, u)
   n = 6
   α = 0.15
   c = 0.12
   du = zeros(3*n)
   for j=1:n
      x, y, z = u[(j - 1)*3 + 1:(j - 1)*3 + 3]
      ω = 1 + 0.02*(j - 1)
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
      du[(j - 1)*3 + 2] = ω*x + α*y + c*(yp1 - 2*y + ym1)
      du[(j - 1)*3 + 3] = 0.1 + z*(x - 8.5)
   end
   return du
end

function rossler2(t, u)
   n = 6
   α = 0.15
   c = 0.12
   du = zeros(3*n)
   for j=1:n
      x, y, z = u[(j - 1)*3 + 1:(j - 1)*3 + 3]
      ω = 1.05 + 0.02*(j - 1)

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
      du[(j - 1)*3 + 2] = ω*x + α*y + c*(yp1 - 2*y + ym1)
      du[(j - 1)*3 + 3] = 0.1 + z*(x - 8.5)
   end
   return du
end

function osc(t, u)
   du = cos.(t).*sin.(u.*[1, 2, 3])
end

function kuramoto(t, u)
   K = 0.1
   N = 9
   du = zeros(N)

   ω = 0.1 .+ 0.02*(1:N)

   for i=1:N
      du[i] = ω[i] + (K/N)*sum([sin(u[j] - u[i]) for j=1:N])
   end

   #du = ω .+ (K/N)*sum(sin(u*ones(1, N)-(ones(N, 1)*u')), dims=1)'

   return du
end

function kuramoto2(t, u)
   K = 0.6
   N = 9
   du = zeros(N)

   ω = 0.1 .+ 0.022*(1:N)

   for i=1:N
      du[i] = ω[i] + (K/N)*sum([sin(u[j] - u[i]) for j=1:N])
   end

   #du = ω .+ (K/N)*sum(sin(u*ones(1, N)-(ones(N, 1)*u')), dims=1)'

   return du
end

function pendulum(t, u)
   x, v, y, u2 = u
   α = 1.0
   du = zeros(4)
   du[1] = v - x
   du[2] = y - sin(α*(x + y))
   du[3] = u2
   du[4] = -y
   return du
end

function pendulum2(t, u)
   x, v, y, u2 = u
   α = 1.05
   du = zeros(4)
   du[1] = v - x
   du[2] = y - sin(α*(x + y) + 0.3)
   du[3] = u2
   du[4] = -y
   return du
end

function harmonic(t, u)
   n = 4
   du = zeros(2*n)
   ω = 0.5 .+ 0.02*(1:n)
   for i=0:n-1
      du[2*i + 1] = -ω[i + 1]*u[2*i + 2]
      du[2*i + 2] = u[2*i + 1]
   end
   return du
end

function harmonic2(t, u)
   n = 4
   du = zeros(2*n)
   ω = [j for j in (0.5 .+ 0.02*(1:n))]
   ω = ω .- 0.01
#   ω[3] = ω[3] + 0.01
   for i=0:n-1
      du[2*i + 1] = -ω[i + 1]*u[2*i + 2]
      du[2*i + 2] = u[2*i + 1]
   end
   return du
end

function colpitts(t, u)
   M = 3
   p1 = 5.0
   p2 = 0.0797
   p3 = [3.0, 3.5, 4.0]
   p4 = 0.6898

   c21 = 0.8
   c32 = 0.9
   c13 = 1.0
   c = [c21, c32, c13]

   du = zeros(3*M)

   for i=0:M-1
      x1, x2, x3 = u[i*3 + 1:i*3 + 3]
      du[3*i + 1] = p1*x2 + c[i + 1]*(u[(3*(i + 1) + 1) % (3*M)] - x1)
      du[3*i + 2] = -p2*(x1 + x3) - p4*x2
      du[3*i + 3] = p3[i + 1]*(x2 + 1 - exp(-x1))
   end
   return du
end

function colpitts2(t, u)
   M = 3
   p1 = 5.0 + 0.04
   p2 = 0.0797 - 0.003
   p3 = [3.0, 3.5, 4.0]
   p4 = 0.6898

   c21 = 0.8
   c32 = 0.9
   c13 = 1.0
   c = [c21, c32, c13]

   du = zeros(3*M)

   for i=0:M-1
      x1, x2, x3 = u[i*3 + 1:i*3 + 3]
      du[3*i + 1] = p1*x2 + c[i + 1]*(u[(3*(i + 1) + 1) % (3*M)] - x1)
      du[3*i + 2] = -p2*(x1 + x3) - p4*x2
      du[3*i + 3] = p3[i + 1]*(x2 + 1 - exp(-x1))
   end
   return du
end

function elegant(t, u)
   x, v, y, u = u
   du = zeros(4)
   k = 1

   du[1] = v + k*v*u^2
   du[2] = -x
   du[3] = u + k*u*v^2
   du[4] = -y

   return du
end

function elegant2(t, u)
   x, v, y, u = u
   du = zeros(4)
   k = 1.2

   du[1] = v + k*v*u^2
   du[2] = -x
   du[3] = u + k*u*v^2
   du[4] = -y

   return du
end

end
