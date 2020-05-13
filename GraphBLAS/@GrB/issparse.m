function s = issparse (G) %#ok<INUSD>
%ISSPARSE always true for any GraphBLAS matrix.
% A GraphBLAS matrix always keeps track of its pattern, even if all
% entries are present.  Thus, issparse (G) is always true for any
% GraphBLAS matrix G; even issparse (full (G)) is true.
%
% To check if all entries are present in G, use this test:
%   (numel (G) == prod (size (G)))
% This is always true for after G = full (G).  The test is false
% if any entry in G is not in its pattern.
%
% See also GrB/ismatrix, GrB/isvector, GrB/isscalar, GrB/isfull, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

s = true ;

