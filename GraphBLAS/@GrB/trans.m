function C = trans (arg1, arg2, arg3, arg4, arg5)
%GRB.TRANS transpose a sparse matrix.
%
% syntax for a new matrix C:                    computation:
% C = GrB.trans (A, desc)                       % C = A'
% C = GrB.trans (Cin, accum, A, desc)           % C = Cin + A'
% C = GrB.trans (Cin, M, A, desc)               % C = Cin ; C<M> = A'
% C = GrB.trans (Cin, M, accum, A, desc)        % C = Cin ; C<M> += A'
%
% GrB.trans computes T=A'.
% T is then accumulated into C via C<M> = accum (C,T), where the accum step
% is computed using GrB.eadd and M can be modified by the descriptor desc.
%
% For complex matrices, GrB.trans computes the array transpose, not the
% matrix (complex conjugate) transpose.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.
%
% See also GrB/transpose, GrB/ctranspose, GrB/conj, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

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
                'usage: C = GrB.trans (Cin, M, accum, A, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;

