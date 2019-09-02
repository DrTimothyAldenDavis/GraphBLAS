function s = tricount (H)
%TRICOUNT count triangles in a gbgraph.
% s = tricount (H) counts the number of triangles in the gbgraph H.
% The gbgraph H must be undirected.

if (isdirected (H))
    error ('input graph must be undirected') ;
end

n = numnodes (H) ;
H = spones (H, 'uint32') ;
C = gb (n, n, 'uint32', gb.format (H)) ;
L = tril (H, -1) ;
U = triu (H, 1) ;

% the methods below are identical, internally in GraphBLAS
if (gb.isbyrow (H))
    % C<L> = L*U'
    C = gb.mxm (C, L, '+.*', L, U, struct ('in1', 'transpose')) ;
else
    % C<U> = L'*U
    C = gb.mxm (C, U, '+.*', L, U, struct ('in0', 'transpose')) ;
end

s = full (double (gb.reduce ('+.int64', C))) ;

