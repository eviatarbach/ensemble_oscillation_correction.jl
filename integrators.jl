module Integrators

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

function leapfrog(f::Function, y0::Array{Float64, 1}, t0::Float64,
                  t1::Float64, h::Float64)
    y = y0
    n = round(Int, (t1 - t0)/h) + 1
    t = t0
    v0 = zeros(length(y0))
    hist = zeros(n, length(y0))
    v = zeros(n, length(y0))

    hist[1, :] = y0
    v[1, :] = v0
    for i in 2:n
      y = hist[i - 1, :] + v[i - 1, :]*h + 0.5*f(t, hist[i - 1, :])*h^2
      hist[i, :] = y
      v[i, :] = v[i - 1, :] + 0.5*(f(t, hist[i - 1, :]) + f(t, hist[i, :]))*h
      t = t0 + i*h
    end
    return hist[2:end, :]
    end

end
