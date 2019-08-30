function G = incidence (H, type)
%INCIDENCE Graph incidence matrix.
% G = incidence (H) is the graph incidence matrix of the gbgraph H.  G is
% GraphBLAS matrix of size n-by-e, if H has n nodes and e edges (not including
% self-edges).  The jth column of has 2 entries: G(s,j) = -1 and G(t,j) = 1,
% where (s,t) is an edge in H.  Self-edges are ignored.
%
% G is a double GraphBLAS matrix by default.  An optional 2nd input argument
% allows the type of G to be determined: G = incidence (G, type).  The type can
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

n = numnodes (H) ;
if (isundirected (H))
    H = tril (H, -1) ;
else
    H = gb.select ('offdiag', H) ;
end
[i, j] = gb.extracttuples (H, struct ('kind', 'zero-based')) ;
e = length (i) ;

k = uint64 (0:e-1)' ;
x = ones (e, 1, type) ;

G = gb.build ([i ; j], [k ; k], [x ; -x], n, e) ;

