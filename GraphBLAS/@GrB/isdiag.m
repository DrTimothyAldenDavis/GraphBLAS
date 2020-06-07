function s = isdiag (G)
%ISDIAG True if the GraphBLAS matrix G is a diagonal matrix.
% isdiag (G) is true if G is a diagonal matrix, and false otherwise.
%
% See also GrB/isbanded.

% FUTURE: this will be much faster when 'gb_bandwidth' is a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% using gb_bandwidth:
% [lo, hi] = gb_bandwidth (G.opaque) ;
% s = (lo == 0) & (hi == 0) ;

G = G.opaque ;
s = (gbnvals (gbselect ('diag', G, 0)) == gbnvals (G)) ;

