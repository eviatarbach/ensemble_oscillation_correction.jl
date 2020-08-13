using LinearAlgebra

using HDF5

D = 4964
N = 17748
n_pcs = 2

r = h5read("r_summed.h5", "r_summed")
r = permutedims(r, [2, 1, 3])
r = reshape(r, D, :)'

C = r'*r/N
u, s, v = svd(C)

h5write("eofs.h5", "eofs", v[:, 1:n_pcs])
h5write("pcs.h5", "pcs", r*v[:, 1:n_pcs])
