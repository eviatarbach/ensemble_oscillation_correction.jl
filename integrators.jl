module Integrators

export rk4_inplace, rk4

function rk4_inplace(f::Function, y0::Array{Float64, 1}, t0::Float64,
                                 t1::Float64, h::Float64)
    y = y0
    n = (t1 - t0)÷h
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
             t1::Float64, h::Float64)
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
    return hist
end

end
