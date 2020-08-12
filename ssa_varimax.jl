module ssa_varimax

export varimax_rotate!

using LinearAlgebra

"""
Varimax rotation as per
   Groth, A. and M. Ghil, 2011: Multivariate singular spectrum analysis and the
      road to phase synchronization, Physical Review E, 84, 036206.
   Groth, A. and M. Ghil, 2015: Monte Carlo Singular Spectrum Analysis (SSA)
      revisited: Detecting oscillator clusters in multivariate datasets, Journal
      of Climate, 28, 7873-7893.

Code based on Matlab implementation by Andreas Groth at
https://www.mathworks.com/matlabcentral/fileexchange/65939-multichannel-singular-spectrum-analysis-varimax-tutorial

Copyright (c) 2018, Andreas Groth
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
function varimax(A::Array{float_type, 3}, reltol=sqrt(eps(float_type)),
                 maxit=1000, normalize=true, G=[]) where {float_type<:AbstractFloat}
   M, D, S = size(A)
   B = A

   B = reshape(B, D*M, S, 1)[:, :, 1]
   D = D*M

   T = diagm(0=>ones(S))

   for iter=1:maxit
      maxTheta = 0
      for i = 1:(S-1)
         for j = (i+1):S
            u = B[:, i].^2 - B[:,j].^2
            v = 2*B[:,i].*B[:,j]
            if M>1
               u = sum(reshape(u, M, :), dims=1)'
               v = sum(reshape(v, M, :), dims=1)'
            end

            usum = sum(u, dims=1);
            vsum = sum(v, dims=1);
            numer = 2*u'*v - 2*usum*vsum/D;
            denom = u'*u - v'*v - (usum^2 - vsum^2)/D;
            theta = atan(numer[1, 1], denom[1, 1])/4;
            maxTheta = max(maxTheta, abs(theta));
            Tij = [cos(theta) -sin(theta); sin(theta) cos(theta)];
            B[:, [i, j]] = B[:, [i, j]] * Tij;
            T[:, [i, j]] = T[:, [i, j]] * Tij;
         end
      end
      if (maxTheta < reltol)
         break
      end
   end

   return B, T
end

function varimax_rotate!(EW, EV, M, D, S)
   EVscaled=EV[:,1:S]*diagm(0=>sqrt.(EW[1:S]))
   EVreshape=reshape(EVscaled,M,D,S)

   _, T = varimax(EVreshape)

   EV[:,1:S]=EV[:,1:S]*T
   EW[1:S]=diag(T'*diagm(0=>EW[1:S])*T)

   EV = EV[:, sortperm(EW, rev=true)]
   EW = sort(EW, rev=true)

   return EW, EV
end

end
