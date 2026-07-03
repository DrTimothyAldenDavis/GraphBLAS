function C = trans (arg1, arg2, arg3, arg4, arg5)
%GHB.TRANS transpose a sparse matrix.
%
% syntax for a new matrix C:                    computation:
% C = GhB.trans (A, desc)                       % C = A'
% C = GhB.trans (Cin, accum, A, desc)           % C = Cin + A'
% C = GhB.trans (Cin, M, A, desc)               % C = Cin ; C<M> = A'
% C = GhB.trans (Cin, M, accum, A, desc)        % C = Cin ; C<M> += A'
%
% in-place syntax:
% GhB.trans (C, A, desc)                        % C = A'
% GhB.trans (C, accum, A, desc)                 % C += A'
% GhB.trans (C, M, A, desc)                     % C<M> = A'
% GhB.trans (C, M, accum, A, desc)              % C<M> += A'
%
% GhB.trans computes T=A'.
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% For complex matrices, GhB.trans computes the array transpose, not the
% matrix (complex conjugate) transpose.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
% For the in-place syntax, the @GhB matrix C is modified in-place.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.
%
% See also GhB/transpose, GhB/ctranspose, GhB/conj, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (nargout == 0)
    switch (nargin)
        case 2
            gbmex_trans (ghb, arg1, arg2) ;
        case 3
            gbmex_trans (ghb, arg1, arg2, arg3) ;
        case 4
            gbmex_trans (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_trans (ghb, arg1, arg2, arg3, arg4, arg5) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GhB.trans (Cin, M, accum, A, desc)') ;
    end
else
    switch (nargin)
        case 1
            [C_opaque, kind] = gbmex_trans (ghb, arg1) ;
        case 2
            [C_opaque, kind] = gbmex_trans (ghb, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_trans (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_trans (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_trans (ghb, arg1, arg2, arg3, arg4, arg5) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GhB.trans (Cin, M, accum, A, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;
end

