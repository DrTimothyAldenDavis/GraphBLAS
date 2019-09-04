function gbgraph_display (G, name, level)
%GBGRAPH_DISPLAY private function for disp and display of a gbgraph.

if (~isempty (name))
    fprintf ('\n%s = \n\n', name) ;
end

n = numnodes (G) ;
[e d] = numedges (G) ;
fprintf ('    GraphBLAS %s graph: %d nodes, %d edges', kind (G), n, e) ;
if (d == 0)
    fprintf (' (no self edges).\n') ;
elseif (d == 1)
    fprintf (' (%d is a self-edge).\n', d) ;
else
    fprintf (' (%d are self-edges).\n', d) ;
end

fprintf ('    adjacency matrix:') ;
disp_helper (G, level) ;

