function C = apply (arg1, arg2, arg3, arg4, arg5, arg6)
%GRB.APPLY apply a unary operator to a matrix.
%
%   C = GrB.apply (op, A)
%   C = GrB.apply (op, A, desc)
%   C = GrB.apply (Cin, accum, op, A, desc)
%   C = GrB.apply (Cin, M, op, A, desc)
%   C = GrB.apply (Cin, M, accum, op, A, desc)
%
% GrB.apply applies a unary operator to the entries in the input matrix A,
% which may be a GraphBLAS or built-in matrix (sparse or full).  See 'help
% GrB.unopinfo' for a list of available unary operators.
%
% The op and A arguments are required.
%
% accum: a binary operator to accumulate the results.  See 'help
% GrB.binopinfo' for available binary operators.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If
% either accum or M is present, then Cin is a required input. If desc.in0
% is 'transpose' then A is transposed before applying the operator, as
% C<M> = accum (C, f(A')) where f(...) is the unary operator.
%
% See also GrB/apply2, GrB/spfun, GrB.unopinfo, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (nargout == 0)
    switch (nargin)
        case 3
            gbmex_apply (ghb, arg1, arg2, arg3) ;
        case 4
            gbmex_apply (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_apply (ghb, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            gbmex_apply (ghb, arg1, arg2, arg3, arg4, arg5, arg6) ;
        otherwise
            error ('GrB:error', ...
                'usage: GhB.apply (C, M, accum, op, A, desc)') ;
    end
else
    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_apply (ghb, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_apply (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_apply (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_apply (ghb, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            [C_opaque, kind] = gbmex_apply (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GhB.apply (Cin, M, accum, op, A, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;
end

