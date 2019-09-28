function Cout = extract (varargin)
%GRB.EXTRACT extract sparse submatrix.
%
% GrB.extract is an interface to GrB_Matrix_extract and
% GrB_Matrix_extract_[TYPE], computing the GraphBLAS expression:
%
%   C<#M,replace> = accum (C, A(I,J)) or accum(C, A(J,I)')
%
% Usage:
%
%   Cout = GrB.extract (Cin, M, accum, A, I, J, desc)
%
% A is a required parameters.  All others are optional, but if M or accum
% appears, then Cin is also required.  If desc.in0 is 'transpose', then
% the description below assumes A = A' is computed first before the
% extraction (A is not changed on output, however).
%
% desc: see 'help GrB.descriptorinfo' for details.
%
% I and J are cell arrays.  I contains 0, 1, 2, or 3 items:
%
%       0:   { }    This is the MATLAB ':', like A(:,J), refering to all m
%                   rows, if A is m-by-n.
%
%       1:   { I }  1D list of row indices, like A(I,J) in MATLAB.
%                   If I is double, then it contains 1-based indices, in
%                   the range 1 to m if A is m-by-n, so that A(1,1) refers
%                   to the entry in the first row and column of A.  If I is
%                   int64 or uint64, then it contains 0-based indices in
%                   the range 0 to m-1, where A(0,0) is the same entry.
%
%       2:  { start,fini }  start and fini are scalars (either double,
%                   int64, or uint64).  This defines I = start:fini in
%                   MATLAB index notation.  Typically, start and fini have
%                   type double and refer to 1-based indexing of A.  int64
%                   or uint64 scalars are treated as 0-based.
%
%       3:  { start,inc,fini } start, inc, and fini are scalars (double,
%                   int64, or uint64).  This defines I = start:inc:fini in
%                   MATLAB notation.  The start and fini are 1-based if
%                   double, 0-based if int64 or uint64.  inc is the same
%                   for any type.
%
%       The J argument is identical, except that it is a list of column
%       indices of A.  If only one cell array is provided, J = {  } is
%       implied, refering to all n columns of A, like A(I,:) in MATLAB
%       notation.  1D indexing of a matrix A, as in C = A(I), is not yet
%       supported.
%
%       If neither I nor J are provided on input, then this implies
%       both I = { } and J = { }, or A(:,:) in MATLAB notation,
%       refering to all rows and columns of A.
%
% Cin: an optional input matrix, containing the initial content of the
%       matrix C.  Cout is the content of C after the assignment is made.
%       If present, Cin argument has size length(I)-by-length(J).
%       If accum is present then Cin is a required input.
%
% accum: an optional binary operator, defined by a string ('+.double') for
%       example.  This allows for Cout = Cin + A(I,J) to be computed.  If
%       not present, no accumulator is used and Cout=A(I,J) is computed.
%       If accum is present then Cin is a required input.
%
% M: an optional mask matrix, the same size as C.
%
% All input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  Cout is returned as a GraphBLAS matrix, by default;
% see 'help GrB/descriptorinfo' for more options.
%
% Example:
%
%   A = sprand (5, 4, 0.5)
%   I = [2 1 5]
%   J = [3 3 1 2]
%   Cout = GrB.extract (A, {I}, {J})
%   C2 = A (I,J)
%   C2 - Cout
%
% See also subsref.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[args, is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    Cout = GrB (gbextract (args {:})) ;
else
    Cout = gbextract (args {:}) ;
end

