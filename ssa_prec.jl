using Serialization
using Statistics

using NetCDF
using HDF5

include("ssa.jl")
using .SSA

M = 60

x = ncread("prec_out.nc", "prec")
#x = x .- mean(x)
x = permutedims(x, [3, 2, 1])

ssa_info = SSA.ssa_decompose(x, M)
r = SSA.ssa_reconstruct(ssa_info, 3:4)

h5write("eig_vals.h5", "eig_vals", ssa_info.eig_vals)
h5write("eig_vecs.h5", "eig_vecs", ssa_info.eig_vecs)
h5write("X.h5", "X", ssa_info.X)
h5write("C.h5", "C", ssa_info.C)
h5write("r.h5", "r", r)
