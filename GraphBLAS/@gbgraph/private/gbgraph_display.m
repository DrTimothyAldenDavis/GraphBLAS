function gbgraph_display (H, name, level)
%GBGRAPH_DISPLAY private function for disp and display of a gbgraph.

if (~isempty (name))
    fprintf ('\n%s = \n\n', name) ;
end

[e d] = numedges (H) ;
fprintf ('    GraphBLAS ') ;
if (isequal (H.graphkind, 'graph'))
    fprintf ('undirected graph') ;
else
    fprintf ('directed graph') ;
end
fprintf (': %d nodes, %d edges', numnodes (H), e) ;
if (d == 0)
    fprintf (' (no self edges).\n') ;
else
    fprintf (' (%d are self-edges).\n', d) ;
end

fprintf ('    adjacency matrix:') ;
disp_helper (H, level) ;

