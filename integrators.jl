module Integrators

using Statistics

using FFTW
#using DSP

export rk4, ks_integrate

function rk4(f::Function, y0::Array{Float64, 1}, t0::Float64,
             t1::Float64, h::Float64; inplace=true)
    y = y0
    n = round(Int, (t1 - t0)/h)
    t = t0
    if ~inplace
        hist = zeros(n, length(y0))
    end
    for i in 1:n
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        if ~inplace
            hist[i, :] = y
        end
        t = t0 + i*h
    end
    if ~inplace
        return hist
    else
        return y
    end
end

function ks_integrate(f, y0::Array{Float64, 1}, t0::Float64,
                  t1::Float64, h::Float64; inplace=true)
    # Based on https://github.com/jswhit/pyks/blob/master/KS.py
    L = 10
    N = 42
    if f == "false"
        diffusion = 1.05
    else
        diffusion = 1.0
    end
    nmax = round(Int, (t1 - t0)/h)

    k = N*fftfreq(N)[1:(N÷2) + 1]/L  # wave numbers

    ik    = im*k                   # spectral derivative operator
    lin   = k.^2 - diffusion*k.^4    # Fourier multipliers for linear term

    x = y0
    if !inplace
        uu = [x]
    end
    # spectral space variable
    xspec = rfft(x)

    # semi-implicit third-order runge kutta update.
    # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
    for i in 1:nmax
        xspec = rfft(x)
        xspec_save = copy(xspec)
        for n in 1:3
            # compute tendency from nonlinear term.
            nlterm = -0.5*ik.*rfft(irfft(xspec, 2*length(xspec) - 2).^2)
            dt = h/(3-(n-1))
            # explicit RK3 step for nonlinear term
            xspec = xspec_save + dt*nlterm
            # implicit trapezoidal adjustment for linear term
            xspec = (xspec+0.5*lin*dt.*xspec_save)./(1.0.-0.5*lin*dt)
        end
        x = irfft(xspec, 2*length(xspec) - 2)
        if !inplace
            push!(uu,x)
        end
    end
    if inplace
        return x
    else
        return hcat(uu...)'
    end

end

end
