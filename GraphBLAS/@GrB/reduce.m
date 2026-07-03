function C = reduce (arg1, arg2, arg3, arg4, arg5)
%GRB.REDUCE reduce a matrix to a scalar.
%
% syntax for a new matrix C:                    computation:
% c = GrB.reduce (op, A, desc)                  % c = op(A)
% c = GrB.reduce (cin, op, A, desc)             % c = cin ; c = op(A)
% c = GrB.reduce (cin, accum, op, A, desc)      % c = cin ; c += op(A)
%
% GrB.reduce reduces a matrix to a scalar, using the given op as a monoid:
%
%   Monoids for real non-logical types: '+', '*', 'max', 'min', 'any'
%   For logical: '|', '&', 'xor', 'eq', 'any'
%   For complex types: '+', '*', 'any'
%   For integer types: 'bitor', 'bitand', 'bitxor', 'bitxnor'
%
% See 'help GrB.monoidinfo' for more details on the available monoids.
%
% The op and A arguments are required.  All others are optional.  The op is
% applied to all entries of the matrix A to reduce them to a single scalar
% result.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
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
            [C_opaque, kind] = gbmex_reduce (ghb, arg1, arg2, arg3, arg4, ...
                arg5) ;
        otherwise
            error ('GrB:error', ...
                'usage: c = GrB.reduce (cin, accum, monoid, A, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;

