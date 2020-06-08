function C = reduce (arg1, arg2, arg3, arg4, arg5)
%GRB.REDUCE reduce a matrix to a scalar.
%
% Usage:
%
%   c = GrB.reduce (monoid, A)
%   c = GrB.reduce (monoid, A, desc)
%   c = GrB.reduce (cin, accum, monoid, A)
%   c = GrB.reduce (cin, accum, monoid, A, desc)
%
% GrB.reduce reduces a matrix to a scalar, using the given monoid.  The
% valid monoids are: '+', '*', 'max', and 'min' for all but the 'logical'
% type, and '|', '&', 'xor', and 'ne' for the 'logical' type.  See 'help
% GrB.monoidinfo' for more details.
%
% TODO fix this description.
%
% The monoid and A arguments are required.  All others are optional.  The
% op is applied to all entries of the matrix A to reduce them to a single
% scalar result.
%
% accum: an optional binary operator (see 'help GrB.binopinfo' for a
% list).
%
% cin: an optional input scalar into which the result can be accumulated
% with c = accum (cin, result).
%
% All input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  c is returned as a GraphBLAS scalar, by default;
% see 'help GrB/descriptorinfo' for more options.
%
% See also GrB.vreduce, GrB/sum, GrB/prod, GrB/max, GrB/min.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (arg1))
    arg1 = arg1.opaque ;
end

if (isobject (arg2))
    arg2 = arg2.opaque ;
end

if (nargin > 2 && isobject (arg3))
    arg3 = arg3.opaque ;
end

if (nargin > 3 && isobject (arg4))
    arg4 = arg4.opaque ;
end

switch (nargin)
    case 2
        [C, k] = gbreduce (arg1, arg2) ;
    case 3
        [C, k] = gbreduce (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbreduce (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbreduce (arg1, arg2, arg3, arg4, arg5) ;
    otherwise
        gb_error ('usage: c = GrB.reduce (cin, accum, monoid, A, desc)') ;
end

if (k == 0)
    C = GrB (C) ;
end

