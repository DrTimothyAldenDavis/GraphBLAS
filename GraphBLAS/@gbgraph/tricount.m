function s = tricount (G)
%TRICOUNT count triangles in a gbgraph.
% s = tricount (G) counts the number of triangles in the gbgraph G.
% The gbgraph G must be undirected.  Self-edges are ignored.
%
% See also gbgraph/ktruss.

if (isdirected (G))
    error ('input graph must be undirected') ;
end

n = numnodes (G) ;
if (n > intmax ('int32'))
    int_type = 'int64' ;
else
    int_type = 'int32' ;
end
G = spones (G, int_type) ;
C = gb (n, n, int_type, gb.format (G)) ;
L = tril (G, -1) ;
U = triu (G, 1) ;

% Inside GraphBLAS, the methods below are identical.  For example, L stored by
% row is the same data structure as U stored by column.

if (gb.isbyrow (G))
    % C<L> = L*U'
    C = gb.mxm (C, L, '+.*', L, U, struct ('in1', 'transpose')) ;
else
    % C<U> = L'*U
    C = gb.mxm (C, U, '+.*', L, U, struct ('in0', 'transpose')) ;
end

s = full (double (gb.reduce ('+.int64', C))) ;

