function Cout = assign (varargin)
%GB.ASSIGN: assign a submatrix into a matrix.
%
% gb.assign is an interface to GrB_Matrix_assign and
% GrB_Matrix_assign_[TYPE], computing the GraphBLAS expression:
%
%   C<#M,replace>(I,J) = accum (C(I,J), A) or accum(C(I,J), A')
%
% where A can be a matrix or a scalar.
%
% Usage:
%
%   Cout = gb.assign (Cin, M, accum, A, I, J, desc)
%   TODO add all valid uses here
%
% Cin and A are required parameters.  All others are optional.
%
% desc: see 'help gb.descriptorinfo' for details.
%
% I and J are cell arrays.  I contains 0, 1, 2, or 3 items:
%
%       0:   { }    This is the MATLAB ':', like C(:,J), refering to all m
%                   rows, if C is m-by-n.
%
%       1:   { I }  1D list of row indices, like C(I,J) in MATLAB.
%                   If I is double, then it contains 1-based indices, in
%                   the range 1 to m if C is m-by-n, so that C(1,1) refers
%                   to the entry in the first row and column of C.  If I is
%                   int64 or uint64, then it contains 0-based indices in
%                   the range 0 to m-1, where C(0,0) is the same entry.
%
%       2:  { start,fini }  start and fini are scalars (either double,
%                   int64, or uint64).  This defines I = start:fini in
%                   MATLAB index notation.  Typically, start and fini have
%                   type double and refer to 1-based indexing of C.  int64
%                   or uint64 scalars are treated as 0-based.
%
%       3:  { start,inc,fini } start, inc, and fini are scalars (double,
%                   int64, or uint64).  This defines I = start:inc:fini in
%                   MATLAB notation.  The start and fini are 1-based if
%                   double, 0-based if int64 or uint64.  inc is the same
%                   for any type.
%
%       The J argument is identical, except that it is a list of column
%       indices of C.  If only one cell array is provided, J = {  } is
%       implied, refering to all n columns of C, like C(I,:) in MATLAB
%       notation.  1D indexing of a matrix C, as in C(I) = A, is not yet
%       supported.
%
%       If neither I nor J are provided on input, then this implies
%       both I = { } and J = { }, or C(:,:) in MATLAB notation,
%       refering to all rows and columns of C.
%
% A: this argument either has size length(I)-by-length(J) (or A' if d.in0
%       is 'transpose'), or it is 1-by-1 for scalar assignment (like
%       C(1:2,1:2)=pi, which assigns the scalar pi to the leading 2-by-2
%       submatrix of C).  For scalar assignment, A must contain an entry;
%       it cannot be empty (for example, the MATLAB A = sparse (0)).
%
% accum: an optional binary operator, defined by a string ('+.double') for
%       example.  This allows for C(I,J) = C(I,J) + A to be computed.  If
%       not present, no accumulator is used and C(I,J)=A is computed.
%
% M: an optional mask matrix, the same size as C.
%
% Cin: a required input matrix, containing the initial content of the
% matrix C.  Cout is the content of C after the assignment is made.
%
% All input matrices may be either GraphBLAS and/or MATLAB matrices, in any
% combination.  Cout is returned as a GraphBLAS matrix.
%
% Example:
%
%   A = sprand (5, 4, 0.5)
%   AT = A'
%   M = sparse (rand (4, 5)) > 0.5
%   Cin = sprand (4, 5, 0.5)
%
%   d.in0 = 'transpose'
%   d.mask = 'complement'
%   Cout = gb.assign (Cin, M, A, d)
%   C2 = Cin
%   C2 (~M) = AT (~M)
%   C2 - sparse (Cout)
%
%   I = [2 1 5]
%   J = [3 3 1 2]
%   B = sprandn (length (I), length (J), 0.5)
%   Cin = sprand (6, 3, 0.5)
%   Cout = gb.assign (Cin, B, {I}, {J})
%   C2 = Cin
%   C2 (I,J) = B
%   C2 - sparse (Cout)
%
% See also gb.subassign, subsasgn

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[args is_gb] = get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbassign (args {:})) ;
else
    Cout = gbassign (args {:}) ;
end

