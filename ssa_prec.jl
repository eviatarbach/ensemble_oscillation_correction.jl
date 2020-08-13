using Serialization
using Statistics

using NetCDF
using HDF5

include("ssa.jl")
using .SSA

M = 60

x = ncread("prec_out.nc", "prec")
x = x .- mean(x)
x = permutedims(x, [3, 2, 1])

ssa_info = SSA.ssa_decompose(x, M)

h5write("EW.h5", "EW", EW)
h5write("EV.h5", "EV", EV)
h5write("X.h5", "X", X)
h5write("C.h5", "C", C)
#serialize(open("prec_ssa", "w"), (EW, EV, X, C))
