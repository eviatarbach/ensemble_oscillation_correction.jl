module DA

export etkf

using Statistics
using LinearAlgebra

"""
Ensemble transform Kalman filter (ETKF)
"""
function etkf(;E, R_inv, inflation, H, y)
    D, m = size(E)

    x_m = mean(E, dims=2)
    X = (E .- x_m)/sqrt(m - 1)

    X = sqrt(inflation)*X

    y_m = H(x_m)
    Y = (vcat([H(E[:, i]) for i=1:m]...) .- y_m)'/sqrt(m - 1)
    Ω = inv(Symmetric(I + Y'*R_inv*Y))
    w = Ω*Y'*R_inv*(y - y_m)'

    E = x_m .+ X*(w .+ sqrt(m - 1)*sqrt(Ω))

    return E
end

end
