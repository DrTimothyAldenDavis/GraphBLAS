function C = apply2 (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.APPLY2 apply a binary operator to a matrix, with scalar binding.
%
%   C = GrB.apply2 (op, A, B)
%   C = GrB.apply2 (op, A, B, desc)
%   C = GrB.apply2 (Cin, accum, op, A, B, desc)
%   C = GrB.apply2 (Cin, M, op, A, B, desc)
%   C = GrB.apply2 (Cin, M, accum, op, A, B, desc)
%
% GrB.apply2 applies a binary operator op(A,B) to a matrix, with one of the
% inputs being the matrix and the other input is bound to a scalar.  See
% 'help GrB.binopinfo'.
%
% The op, A, and B arguments are required.  One of A or B must be a scalar
% with a single entry.
%
% accum: a binary operator to accumulate the results.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If
% either accum or M is present, then Cin is a required input.  If B is the
% scalar and desc.in0 is 'transpose' then A is transposed before applying
% the operator.  If A is the scalar and desc.in1 is 'transpose.', then the
% input matrix B is transposed before applying the operator.
%
% See also GrB/apply, GrB/spfun, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (nargout == 0)
    switch (nargin)
        case 4
            gbmex_apply2 (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_apply2 (ghb, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            gbmex_apply2 (ghb, arg1, arg2, arg3, arg4, arg5, arg6) ;
        case 7
            gbmex_apply2 (ghb, arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
        otherwise
            error ('GrB:error', ...
                'usage: GhB.apply2 (C, M, accum, op, A, B, desc)') ;
    end
else
    switch (nargin)
        case 3
            [C_opaque, kind] = gbmex_apply2 (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_apply2 (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_apply2 (ghb, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_apply2 (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_apply2 (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GhB.apply2 (Cin, M, accum, op, A, B, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;
end

