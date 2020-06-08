function I = subsindex (G)
%SUBSINDEX subscript index from GraphBLAS matrix
% I = subsindex (G) is used when the GraphBLAS matrix G is used to
% index into a non-GraphBLAS matrix A, for A(G).
%
% See also GrB/subsref, GrB/subsasgn.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

I = gb_subsindex (G.opaque, 1) ;

