function C = reduce (arg1, arg2, arg3, arg4, arg5)
%GRB.REDUCE reduce a matrix to a scalar.
%
%   c = GrB.reduce (monoid, A)
%   c = GrB.reduce (monoid, A, desc)
%   c = GrB.reduce (cin, accum, monoid, A)
%   c = GrB.reduce (cin, accum, monoid, A, desc)
%
% GrB.reduce reduces a matrix to a scalar, using the given monoid:
%
%   Monoids for real non-logical types: '+', '*', 'max', 'min', 'any'
%   For logical: '|', '&', 'xor', 'eq', 'any'
%   For complex types: '+', '*', 'any'
%   For integer types: 'bitor', 'bitand', 'bitxor', 'bitxnor'
%
% See 'help GrB.monoidinfo' for more details on the available monoids.
%
% The monoid and A arguments are required.  All others are optional.  The
% op is applied to all entries of the matrix A to reduce them to a single
% scalar result.
%
% accum: an optional binary operator.
%       See 'help GrB.binopinfo' for available binary operators.
%
% cin: an optional input scalar into which the result can be accumulated
% with c = accum (cin, result).
%
% See also GrB.vreduce, GrB/sum, GrB/prod, GrB/max, GrB/min, GrB.monoidinfo,
% GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

switch (nargin)
    case 2
        [C_opaque, kind] = gbmex_reduce (ghb, arg1, arg2) ;
    case 3
        [C_opaque, kind] = gbmex_reduce (ghb, arg1, arg2, arg3) ;
    case 4
        [C_opaque, kind] = gbmex_reduce (ghb, arg1, arg2, arg3, arg4) ;
    case 5
        [C_opaque, kind] = gbmex_reduce (ghb, arg1, arg2, arg3, arg4, arg5) ;
end

C = gb_mexfunction_result (ghb, C_opaque, kind) ;

