function C = kronecker (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.KRONECKER sparse Kronecker product.
%
%   C = GrB.kronecker (op, A, B, desc)
%   C = GrB.kronecker (Cin, accum, op, A, B, desc)
%   C = GrB.kronecker (Cin, M, op, A, B, desc)
%   C = GrB.kronecker (Cin, M, accum, op, A, B, desc)
%
% GrB.kronecker computes the Kronecker product, T=kron(A,B), using the
% given binary operator op, in place of the conventional '*' operator for
% the built-in kron.  See also C = kron (A,B), which uses the
% default semiring operators if A and/or B are GrB matrices.
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% See also kron, GrB/kron, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

switch (nargin)
    case 3
        [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3) ;
    case 4
        [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3, arg4) ;
    case 5
        [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C_opaque, kind] = gbmex_kronecker (ghb, arg1, arg2, arg3, arg4, arg5, arg6, ...
            arg7) ;
end

C = gb_mexfunction_result (C_opaque, kind) ;

