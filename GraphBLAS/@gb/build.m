function C = build (varargin)
%GB.BUILD construct a GraphBLAS sparse matrix from a list of entries.
%
% Usage
%
%   C = gb.build (I, J, X, m, n, dup, type, desc)
%
% gb.build constructs an m-by-n GraphBLAS sparse matrix C from a list of
% entries, analogous to A = sparse (I, J, X, m, n) to construct a MATLAB
% sparse matrix A.
%
% If not present or empty, m defaults to the largest row index in the
% list I, and n defaults to the largest column index in the list J.  dup
% defaults to '+', which gives the same behavior as the MATLAB sparse
% function: duplicate entries are added together.
%
% dup is a string that defines a binary function; see 'help gb.binopinfo'
% for a list of available binary operators.  The dup operator need not be
% associative.  If two entries in [I,J,X] have the same row and column
% index, the dup operator is applied to assemble them into a single
% entry.  Suppose (i,j,x1), (i,j,x2), and (i,j,x3) appear in that order
% in [I,J,X], in any location (the arrays [I J] need not be sorted, and
% so these entries need not be adjacent).  That is, i = I(k1) = I(k2) =
% I(k3) and j = J(k1) = J(k2) = J(k3) for some k1 < k2 < k3.  Then C(i,j)
% is computed as follows, in order:
%
%   x = X (k1) ;
%   x = dup (x, X (k2)) ;
%   x = dup (x, X (k3)) ;
%   C (i,j) = x ;
%
% For example, if the dup operator is '1st', then C(i,j)=X(k1) is set,
% and the subsequent entries are ignored.  If dup is '2nd', then
% C(i,j)=X(k3), and the preceding entries are ignored.
%
% type is a string that defines the type of C (see 'help gb' for a list
% of types).  The type need not be the same type as the dup operator
% (unless one has a type of 'complex', in which case both must be
% 'complex').  If the type is not specified, it defaults to the type of
% X.
%
% The integer arrays I and J may be double, in which case they contain
% 1-based indices, in the range 1 to the dimension of the matrix.  This
% is the same behavior as the MATLAB sparse function.  They may instead
% be int64 or uint64 arrays, in which case they are treated as 0-based.
% Entries in I are the range 0 to m-1, and J are in the range 0 to n-1.
% If I, J, and X are double, the following examples construct the same
% MATLAB sparse matrix S:
%
%   S = sparse (I, J, X) ;
%   S = gb.build (I, J, X, struct ('kind', 'sparse')) ;
%   S = double (gb.build (I, J, X)) ;
%   S = double (gb.build (uint64(I)-1, uint64(J)-1, X)) ;
%
% Using uint64 integers for I and J is faster and uses less memory.  I
% and J need not be in any particular order, but gb.build is fastest if I
% and J are provided in column-major order.
%
% Note: S = sparse (I,J,X) allows either I or J, and X to be scalars.
% This feature is not supported in gb.build.  All three arrays must be
% the same size.
%
% See also sparse, find, gb.extracttuples.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[args, is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    C = gb (gbbuild (args {:})) ;
else
    C = gbbuild (args {:}) ;
end

