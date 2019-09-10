module Integrators

using FFTW, Statistics

export rk4_inplace, rk4

function rk4_inplace(f::Function, y0::Array{Float64, 1}, t0::Float64,
                     t1::Float64, h::Float64)
    y = y0
    n = round(Int, (t1 - t0)/h)
    t = t0
    for i in 1:n
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t0 + i*h
    end
    return y
end

function rk4(f::Function, y0::Array{Float64, 1}, t0::Float64,
             t1::Float64, h::Float64, outfreq::Int64=1)
    y = y0
    n = round(Int, (t1 - t0)/h)
    t = t0
    hist = zeros(n, length(y0))
    for i in 1:n
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        hist[i, :] = y
        t = t0 + i*h
    end
    return hist[outfreq:outfreq:end, :]
end

function ks_integrate(f, y0::Array{Float64, 1}, t0::Float64,
                  t1::Float64, h::Float64; inplace=true)
    Q = 100
    if f == "true"
        N = 64
    else
        N = 60
    end
    nmax = round(Int,t1/h)
    x = N*(1:Q) / Q
    u = y0
    v = fft(u)
    k = 2π/N*(0.0:Q-1)
    L = (k.^2 - k.^4) # Fourier Multiples
    E =  exp.(h*L)
    E2 = exp.(h*L/2)
    M = 16 # No. of points for complex means
    r = exp.(im*π*((1:M) .-0.5)/M)
    LR = h*L*ones(M)' + ones(Q)*r'
    QQ = h*real.(mean((exp.(LR/2).-1)./LR, dims=2))[:]
    f1 = h*real(mean((-4 .-LR+exp.(LR).*(4 .-3*LR+ LR.^2))./LR.^3,dims=2))[:]
    f2 = h*real(mean((2 .+LR+exp.(LR).*(-2 .+LR))./LR.^3,dims=2))[:]
    f3 = h*real(mean((-4 .-3*LR-LR.^2+exp.(LR).*(4 .-LR))./LR.^3,dims=2))[:]
    g = -0.5im*k
    if !inplace
        uu = [u]
    end

    T = plan_fft(v)
    Ti = plan_ifft(v)
    T! = plan_fft!(v)
    Ti! = plan_ifft!(v)
    a = Complex.(zeros(Q))
    b = Complex.(zeros(Q))
    c = Complex.(zeros(Q))
    Nv = Complex.(zeros(Q))
    Na = Complex.(zeros(Q))
    Nb = Complex.(zeros(Q))
    Nc = Complex.(zeros(Q))

    for n = 1:nmax
                          Nv .= g .* (T*real(Ti*v).^2) #.+ cc
      @.  a  =  E2*v + QQ*Nv
                         Na .= g .* (T!*real(Ti!*a).^2) #.+ cc
      @. b  =  E2*v + QQ*Na
                          Nb .= g.* (T!*real(Ti!*b).^2) #.+ cc
      @. c  =  E2*a + QQ*(2Nb-Nv)
      Nc .= g.* (T!*real(Ti!*c).^2) #.+ cc
      @. v =  E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3

      u = real(Ti*v)
      if !inplace
          push!(uu,u)
      end
    end
    if inplace
        return u
    else
        return hcat(uu...)'
    end
end

end
