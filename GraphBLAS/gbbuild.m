function A = gbbuild (I, J, X, m, n, dup, type)
%GBBUILD construct a SuiteSparse:GraphBLAS sparse matrix from a list of entries.
%
% Usage
%   A = gbbuild (I, J, X, m, n, dup, type)
%
% gbbuild constructs an m-by-n GraphBLAS sparse matrix from a list of entries,
% analogous to A = sparse (I, J, X, m, n) to construct a MATLAB sparse matrix.
%
% If not present or empty, m defaults to the largest row index in the list I,
% and n defaults to the largest column index in the list J.  dup defaults to
% '+', which gives the same behavior as the MATLAB sparse function: duplicate
% entries are added together.
%
% dup is a string that defines a binary function; see 'help gbbinop' for a list
% of available binary operators.  The dup operator need not be associative.  If
% two entries in [I J X] have the same row and column index, the dup operator
% is applied to assemble them into a single entry.  Suppose (i,j,x1), (i,j,x2),
% and (i,j,x3) appear in that order in [I J X], in any location (the arrays [I
% J] need not be sorted, and so these entries need not be adjacent).  That is,
% i = I(k1) = I(k2) = I(k3) and j = J(k1) = J(k2) = J(k3) for some k1 < k2 <
% k3.  Then A(i,j) is computed as follows, in order:
%
%   x = X (k1) ;
%   x = dup (x, X (k2)) ;
%   x = dup (x, X (k3)) ;
%   A (i,j) = x ;
%
% For example, if the dup operator is '1st', then A(i,j)=X(k1) is set, and
% the subsequent entries are ignored.  If dup is '2nd', then A(i,j)=X(k3),
% and the preceding entries are ignored.
%
% type is a string that defines the type of A (see 'help gb' for a list of
% types).  The type need not be the same type as the dup operator (unless
% one has a type of 'complex', in which case both must be 'complex').  If the
% type is not specified, it defaults to the type of X.
%
% The integer arrays I and J may be double, in which case they contain 1-based
% indices, in the range 1 to the dimension of the matrix.  This is the same
% behavior as the MATLAB sparse function.  They may instead be uint64 arrays,
% in which case they are treated as 0-based.  Entries in I are the range 0 to
% m-1, and J are in the range 0 to n-1.  If I, J, and X are double, the
% following examples construct the same MATLAB sparse matrix:
%
%   A = sparse (I, J, X) ;
%   A = gbsparse (gbbuild (I, J, X)) ;
%   A = gbsparse (gbbuild (uint64(I)-1, uint64(J)-1, X)) ;
%
% Using uint64 integers for I and J is faster and uses less memory.  I and J
% need not be in any particular order, but gbbuild is fastest if I and J are
% provided in column-major order.
%
% See also sparse (with 3 or more input arguments), gbsparse.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbbuild mexFunction not found; use gbmake to compile GraphBLAS') ;
