function C = ktruss (G, k)
%KTRUSS find the k-truss of a gbgraph G.
% C = ktruss (G, k) finds the k-truss of a gbgraph G.  G must be symmetric with
% no self-edges, and any edge weights of G are ignored.  The ktruss C is a
% graph consisting of a subset of the edges of G.  Each edge in C is part of at
% least k-2 triangles in G, where a triangle is a set of 3 unique nodes that
% form a clique.  The pattern of C is the k-truss, and the edge weights of C
% are the support of each edge.  That is, C(i,j) = nt if the edge (i,j) is part
% of nt triangles in C.  All edges in C have a support of at least nt >= k-2.
% The total number of triangles in C is sum(C,'all')/6.  C is returned as a
% symmetric graph with no self-edges.  If k is not present, it defaults to 3.
%
% To compute a sequence of k-trusses, a k1-truss can be efficiently used to
% construct another k2-truss with k2 > k1.
%
% Example:
%
%   load west0479 ;
%   G = underlying (pruneself (gbgraph (west0479))) ;
%   C3 = ktruss (G, 3) ;
%   C4a = ktruss (G, 4) ;
%   C4b = ktruss (C3, 4) ;          % this is faster
%   assert (isequal (C4a, C4b)) ;
%
% See also gbgraph/tricount.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% check inputs
if (nargin < 2)
    k = 3 ;
end
if (k < 3)
    error ('k-truss defined only for k >= 3') ;
end
if (isdirected (G))
    error ('k-truss not defined for a directed graph G') ;
end
if (nself (G) > 0)
    error ('k-truss not defined for a graph with self-edges') ;
end

% initializations
n = numnodes (G) ;
if (n > intmax ('int32'))
    C = spones (G, 'int64') ;
else
    C = spones (G, 'int32') ;
end
lastnz = nnz (C) ;

while (1)
    % C<C> = C*C using the plus-and semiring, then drop any < k-2.
    C = gb.select ('>=thunk', gb.mxm (C, C, '+.&', C, C), k-2) ;
    nz = nnz (C) ;
    if (lastnz == nz)
        % quit when the matrix does not change
        break ;
    end
    lastnz = nz ;
end

C = gbgraph (C, 'undirected') ;

