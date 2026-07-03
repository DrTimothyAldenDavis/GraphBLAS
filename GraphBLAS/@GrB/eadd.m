function C = eadd (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.EADD sparse matrix 'addition'.
%
% syntax for a new matrix C:                    computation:
% C = GrB.eadd (op, A, B, desc)                 % C = op(A,B)
% C = GrB.eadd (Cin, accum, op, A, B, desc)     % C = Cin + op(A,B)
% C = GrB.eadd (Cin, M, op, A, B, desc)         % C = Cin ; C<M> = op(A,B)
% C = GrB.eadd (Cin, M, accum, op, A, B, desc)  % C = Cin ; C<M> += op(A,B)
%
% GrB.eadd computes the element-wise 'addition' T=A+B, using any binary op
% (shown as op(A,B) in the computations listed above).  The result T has the
% pattern of the union of A and B. The operator is used where A(i,j) and B(i,j)
% are present.  Otherwise the entries in A and B are copied directly into T:
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%   elseif (A(i,j) is present but B(i,j) is not)
%       T(i,j) = A(i,j)
%   elseif (B(i,j) is present but A(i,j) is not)
%       T(i,j) = B(i,j)
%
% T is then accumulated into C via C<M> = accum (C,T), where the accum step
% is computed using GrB.eadd and M can be modified by the descriptor desc.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.  If desc.in1
% is 'transpose', then the input matrix B is transposed before applying the
% operator.
%
% See also GrB.emult, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

    switch (nargin)
        case 3
            [C_opaque, kind] = gbmex_eadd (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_eadd (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_eadd (ghb, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            [C_opaque, kind] = gbmex_eadd (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_eadd (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GrB.eadd (Cin, M, accum, op, A, B, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;

