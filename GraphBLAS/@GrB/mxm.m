function C = mxm (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GRB.MXM sparse matrix-matrix multiplication.
%
% GrB.mxm computes C<M> = accum (C, A*B) using a given semiring.
%
% Usage:
%
%   C = GrB.mxm (semiring, A, B)
%   C = GrB.mxm (semiring, A, B, desc)
%
%   C = GrB.mxm (Cin, accum, semiring, A, B)
%   C = GrB.mxm (Cin, accum, semiring, A, B, desc)
%
%   C = GrB.mxm (Cin, M, semiring, A, B)
%   C = GrB.mxm (Cin, M, semiring, A, B, desc)
%
%   C = GrB.mxm (Cin, M, accum, semiring, A, B)
%   C = GrB.mxm (Cin, M, accum, semiring, A, B, desc)
%
% Not all inputs are required.
%
% Cin is an optional input matrix.  If Cin is not present or is an empty
% matrix (Cin = [ ]) then it is implicitly a matrix with no entries, of
% the right size (which depends on A, B, and the descriptor).  Its type
% is the output type of the accum operator, if it is present; otherwise,
% its type is the type of the additive monoid of the semiring.
%
% M is the optional mask matrix.  If not present, or if empty, then no
% mask is used.  If present, M must have the same size as C.
%
% If accum is not present, then the operation becomes C<...> = A*B.
% Otherwise, accum (C,A*B) is computed.  The accum operator acts like a
% sparse matrix addition (see GrB.eadd).
%
% The semiring is a required string defining the semiring to use, in the
% form 'add.mult.type', where '.type' is optional.  For example,
% '+.*.double' is the conventional semiring for numerical linear algebra,
% used in MATLAB for C=A*B when A and B are double.  If A or B are double
% complex, then the '+.*.double complex' semiring is used. GraphBLAS has
% many more semirings it can use.  See 'help GrB.semiringinfo' for more
% details.
%
% A and B are the input matrices.  They are transposed on input if
% desc.in0 = 'transpose' (which transposes A), and/or desc.in1 =
% 'transpose' (which transposes B).
%
% The descriptor desc is optional.  If not present, all default settings
% are used.  Fields not present are treated as their default values.  See
% 'help GrB.descriptorinfo' for more details.
%
% All input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix, by default;
% see 'help GrB/descriptorinfo' for more options.
%
% Examples:
%
%   A = sprand (4,5,0.5) ;
%   B = sprand (5,3,0.5) ;
%   C = GrB.mxm ('+.*', A, B) ;
%   norm (C-A*B,1)
%   E = sprand (4,3,0.7) ;
%   M = logical (sprand (4,3,0.5)) ;
%   C2 = GrB.mxm (E, M, '+', '+.*', A, B) ;
%   C3 = E ; AB = A*B ; C3 (M) = C3 (M) + AB (M) ;
%   norm (C2-C3,1)
%
% See also GrB.descriptorinfo, GrB.add, GrB/mtimes.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (arg1))
    arg1 = arg1.opaque ;
end

if (isobject (arg2))
    arg2 = arg2.opaque ;
end

if (isobject (arg3))
    arg3 = arg3.opaque ;
end

if (nargin > 3 && isobject (arg4))
    arg4 = arg4.opaque ;
end

if (nargin > 4 && isobject (arg5))
    arg5 = arg5.opaque ;
end

if (nargin > 5 && isobject (arg6))
    arg6 = arg6.opaque ;
end

switch (nargin)
    case 3
        [C, k] = gbmxm (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbmxm (arg1, arg2, arg3, arg4) ;
    case 5
        [C, k] = gbmxm (arg1, arg2, arg3, arg4, arg5) ;
    case 6
        [C, k] = gbmxm (arg1, arg2, arg3, arg4, arg5, arg6) ;
    case 7
        [C, k] = gbmxm (arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
    otherwise
        gb_error ('usage: C = GrB.mxm (Cin, M, accum, semiring, A, B, desc)') ;
end

if (k == 0)
    C = GrB (C) ;
end

