module SSA_CP

using LinearAlgebra

include("ssa.jl")

using .SSA

function precomp(C, M, D, mode)
   C_conds = Array{Array{Float64, 2}, 1}()

   for offset=1:M-1

      if mode == 'f'
         C_11 = reshape(reshape(C, M, D, M, D)[1:end-offset, :, 1:end-offset, :], (M - offset)*D, (M - offset)*D)
         C_21 = reshape(reshape(C, M, D, M, D)[end-offset+1:end, :, 1:end-offset, :], :, (M - offset)*D)
      elseif mode == 'b'
         C_11 = reshape(reshape(C, M, D, M, D)[offset+1:end, :, offset+1:end, :], (M - offset)*D, (M - offset)*D)
         C_21 = reshape(reshape(C, M, D, M, D)[1:offset, :, offset+1:end, :], :, (M - offset)*D)
      end

      C_11 += 1e-8*I  # To improve conditioning
      C_11_inv = inv(C_11)

      push!(C_conds, C_21*C_11_inv)
   end
   return C_conds
end

function transform(x::Array{float_type, 2}, M::Integer, mode, C_conds) where {float_type<:AbstractFloat}
   # Method from "Singular Spectrum Analysis With Conditional Predictions for
   # Real-Time State Estimation and Forecasting", Ogrosky et al. (2019)

   N, D = size(x)

   N′ = N-M+1
   X = zeros(float_type, N′, D*M)

   for d = 1:D
      for i = 1:N′
          X[i, 1+M*(d-1):M*d] = x[i:i+M-1, d]
      end
   end

   Xp = zeros(N, M*D)

   if mode == 'f'
      Xp[1:N-M+1, :] = X

      X_end = reshape(X[end, :], M, D)
      indices = (N-M+2):N
   elseif mode == 'b'
      Xp[M:end, :] = X

      X_end = reshape(X[1, :], M, D)
      indices = M-1:-1:1
   end

   Xp = reshape(Xp, N, M, D)

   for k=indices
      if mode == 'f'
         offset = k - (N - M + 1)

         # Fill in upper diagonal with known values
         Xp[k, 1:(end-offset), :] = X_end[(offset + 1):end, :]

         C_cond = C_conds[offset]

         # Fill in unknown values
         Xp[k, (end-offset+1):end, :] = C_cond*reshape(Xp[k, 1:end-offset, :], D*(M - offset))
      elseif mode == 'b'
         offset = M - k

         Xp[k, offset+1:end, :] = X_end[1:end-offset, :]

         C_cond = C_conds[offset]

         Xp[k, 1:offset, :] = C_cond*reshape(Xp[k, (offset+1):end, :], D*(M - offset))
      end
   end

   Xp = reshape(Xp, N, M*D)

   return Xp
end

end
