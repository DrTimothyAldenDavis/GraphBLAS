function C = emult (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.EMULT sparse element-wise 'multiplication'.
%
% syntax for a new matrix C:                    computation:
% C = GrB.emult (op, A, B, desc)                % C = op(A,B)
% C = GrB.emult (Cin, accum, op, A, B, desc)    % C = Cin + op(A,B)
% C = GrB.emult (Cin, M, op, A, B, desc)        % C = Cin ; C<M> = op(A,B)
% C = GrB.emult (Cin, M, accum, op, A, B, desc) % C = Cin ; C<M> += op(A,B)
%
% GrB.emult computes the element-wise 'multiplication' T=A.*B, using any binary
% op (shown as op(A,B) in the computations listed above).  The result T has the
% pattern of the intersection of A and B. The operator is used where A(i,j) and
% B(i,j) are present.  Otherwise the entry does not appear in T.
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
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
% See also GrB.eadd, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

    switch (nargin)
        case 3
            [C_opaque, kind] = gbmex_emult (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_emult (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_emult (ghb, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            [C_opaque, kind] = gbmex_emult (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_emult (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GrB.emult (Cin, M, accum, op, A, B, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;

