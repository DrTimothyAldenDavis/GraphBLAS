function C = trans (arg1, arg2, arg3, arg4, arg5)
%GRB.TRANS transpose a sparse matrix.
%
%   C = GrB.trans (A)
%   C = GrB.trans (A, desc)
%   C = GrB.trans (Cin, accum, A, desc)
%   C = GrB.trans (Cin, M, A, desc)
%   C = GrB.trans (Cin, M, accum, A, desc)
%
% The descriptor is optional.  If desc.in0 is 'transpose', then C<M>=A or
% C<M>=accum(C,A) is computed, since the default behavior is to transpose
% the input matrix.
%
% All input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix, by default;
% see 'help GrB/descriptorinfo' for more options.
%
% For complex matrices, GrB.trans computes the array transpose, not the
% matrix (complex conjugate) transpose.
%
% See also GrB/transpose, GrB/ctranspose, GrB/conj.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (arg1))
assert(false) ;
    arg1 = arg1.opaque ;
end

if (nargin > 1 && isobject (arg2))
assert(false) ;
    arg2 = arg2.opaque ;
end

if (nargin > 2 && isobject (arg3))
assert(false) ;
    arg3 = arg3.opaque ;
end

if (nargin > 3 && isobject (arg4))
assert(false) ;
    arg4 = arg4.opaque ;
end

switch (nargin)
    case 1
        [C, k] = gbtrans (arg1) ;
    case 2
assert(false) ;
        [C, k] = gbtrans (arg1, arg2) ;
    case 3
        [C, k] = gbtrans (arg1, arg2, arg3) ;
    case 4
        [C, k] = gbtrans (arg1, arg2, arg3, arg4) ;
    case 5
assert(false) ;
        [C, k] = gbtrans (arg1, arg2, arg3, arg4, arg5) ;
    otherwise
assert(false) ;
        error ('usage: C = GrB.trans (Cin, M, accum, A, desc)') ;
end

if (k == 0)
    C = GrB (C) ;
end

