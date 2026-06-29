function C = eunion (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
%GRB.EUNION sparse matrix union.
%
%   C = GrB.eunion (op, A, a, B, b)
%   C = GrB.eunion (op, A, a, B, b, desc)
%   C = GrB.eunion (Cin, accum, op, A, a, B, b, desc)
%   C = GrB.eunion (Cin, M, op, A, a, B, b, desc)
%   C = GrB.eunion (Cin, M, accum, op, A, a, B, b, desc)
%
% GrB.euion computes the element-wise 'addition' T=A+B.  The result T has
% the pattern of the union of A and B. The operator is used for all entries
% in C(i,j), where a and b are scalars:
%
%   if (A(i,j) and B(i,j) is present)
%       T(i,j) = op (A(i,j), B(i,j))
%   elseif (A(i,j) is present but B(i,j) is not)
%       T(i,j) = op (A(i,j), b)
%   elseif (B(i,j) is present but A(i,j) is not)
%       T(i,j) = op (a, B(i,j))
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% Cin, M, accum, and the descriptor desc are the same as all other
% GrB.methods; see GrB.mxm and GrB.descriptorinfo for more details.  For
% the binary operator, see GrB.binopinfo.
%
% See also GrB.emult, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

    switch (nargin)
        case 5
            [C_opaque, kind] = gbmex_eunion (ghb, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_eunion (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_eunion (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
        case 8
            [C_opaque, kind] = gbmex_eunion (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7, arg8) ;
        case 9
            [C_opaque, kind] = gbmex_eunion (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7, arg8, arg9) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GrB.eunion (Cin, M, accum, op, A, a, B, b, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;

