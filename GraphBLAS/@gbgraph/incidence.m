function C = incidence (G, type)
%INCIDENCE Graph incidence matrix.
% C = incidence (G) is the graph incidence matrix of the gbgraph G.  C is
% GraphBLAS matrix of size n-by-e, if G has n nodes and e edges (not including
% self-edges).  The jth column of has 2 entries: C(s,j) = -1 and C(t,j) = 1,
% where (s,t) is an edge in G.  Self-edges are ignored.
%
% C is a double GraphBLAS matrix by default.  An optional 2nd input argument
% allows the type of C to be determined: C = incidence (G, type).  The type can
% be 'double', 'single', 'int8', 'int16', 'int32', or 'int64'.
%
% See also graph/incidence, digraph/incidence.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin < 2)
    type = 'double' ;
end

if (~gb.issigned (type))
    error ('type must be a signed type') ;
end

n = numnodes (G) ;
if (isundirected (G))
    G = tril (G, -1) ;
else
    G = gb.select ('offdiag', G) ;
end
[i, j] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
e = length (i) ;

k = uint64 (0:e-1)' ;
x = ones (e, 1, type) ;

C = gb.build ([i ; j], [k ; k], [x ; -x], n, e) ;

