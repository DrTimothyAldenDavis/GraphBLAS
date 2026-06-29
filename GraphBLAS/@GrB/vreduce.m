function C = vreduce (arg1, arg2, arg3, arg4, arg5, arg6)
%GRB.VREDUCE reduce a matrix to a vector.
%
%   C = GrB.vreduce (monoid, A)
%   C = GrB.vreduce (monoid, A, desc)
%   C = GrB.vreduce (Cin, M, monoid, A)
%   C = GrB.vreduce (Cin, M, monoid, A, desc)
%   C = GrB.vreduce (Cin, accum, monoid, A)
%   C = GrB.vreduce (Cin, accum, monoid, A, desc)
%   C = GrB.vreduce (Cin, M, accum, monoid, A)
%   C = GrB.vreduce (Cin, M, accum, monoid, A, desc)
%
% The monoid and A arguments are required.  All others are optional.
%
%   Monoids for real non-logical types: '+', '*', 'max', 'min', 'any'
%   For logical: '|', '&', 'xor', 'eq', 'any'
%   For complex types: '+', '*', 'any'
%   For integer types: 'bitor', 'bitand', 'bitxor', 'bitxnor'
%
% See 'help GrB.monoidinfo' for more details on the available monoids.
%
% By default, each row of A is reduced to a scalar.  If Cin is not present,
% C (i) = reduce (A (i,:)).  In this case, Cin and C are column vectors of
% size m-by-1, where A is m-by-n.  If desc.in0 is 'transpose', then A.' is
% reduced to a column vector; C (j) = reduce (A (:,j)).  In this case, Cin
% and C are column vectors of size n-by-1, if A is m-by-n.
%
% accum: an optional binary operator.
%       See 'help GrB.binopinfo' for available binary operators.
%
% See also GrB.reduce, GrB/sum, GrB/prod, GrB/max, GrB/min, GrB.monoidinfo,
% GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_vreduce (ghb, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_vreduce (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_vreduce (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_vreduce (ghb, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_vreduce (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GrB.vreduce (Cin, M, accum, monoid, A, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;

